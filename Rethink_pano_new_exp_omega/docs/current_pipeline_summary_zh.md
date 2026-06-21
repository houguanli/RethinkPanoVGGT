# Rethink Pano VGGT-Omega 当前管线总结

## 1. 对照范围

本文基于以下内容整理：

- 初始 guide：初始提交 `8747f64` 中的 `README_LUNA_OMEGA.md`。
- 当前代码：`vggt_omega/`、`training/`、`scripts/`、`configs/` 和 `tests/`。
- 当前分支：`omega-single-pano`。

初始 guide 的目标是把 b1 版本的 LUNA 全景增强移植到 VGGT-Omega：将等距柱状全景图拆成虚拟透视窗口，在不破坏原始 Omega 权重兼容性的前提下，通过轻量残差模块补充跨窗口全景信息和已知虚拟相机信息。

当前项目已经从“模型结构移植和 smoke test”扩展为一套包含数据读取、全景采样、深度监督、训练、检查点、单全景重建和评估工具的完整实验管线。

## 2. 当前管线总览

```text
ERP 全景 RGB + ERP 深度
        |
        v
数据集解码、无效深度处理、单/多全景分组
        |
        v
PanoWindowSampler
  - ERP -> 多个虚拟 pinhole 窗口
  - 生成 yaw / pitch / FoV / rotation
  - 生成 patch 的球面方向、ERP 坐标、全局 patch ID、接缝标记
        |
        v
VGGT-Omega Aggregator
  - 原始 frame attention
  - 原始 inter-frame global/register attention
  - pano-global token
  - LUNA-Patch 跨窗口 patch-bank 残差
  - LUNA-Camera 已知虚拟相机残差
        |
        v
DenseHead / CameraHead
  - 窗口 Z-depth 与置信度
  - 可选 pose encoding
        |
        v
监督目标构造
  - ERP 存储深度 -> radial range
  - masked bilinear 采样到虚拟窗口
  - radial range -> pinhole Z-depth
        |
        v
log-depth 损失 + 可选相机损失
        |
        v
LUNA / DenseHead / CameraHead 的选择性微调
        |
        v
窗口深度、ERP splat、已知窗口点云、预测相机点云、尺度标定与 Sim3 评估
```

## 3. 各阶段工作流程

### 3.1 数据输入与分组

当前支持两类数据：

1. `PanoVKittiOmegaDataset`
   - 读取 VKitti 风格目录中的 ERP RGB、ERP 深度和 `pano_meta.json`。
   - 从元数据恢复深度缩放比例和全景相机位置。
   - 支持单全景、固定邻域和可变邻域采样。
   - 邻域模式以 anchor pano 为第一项，并按空间位置构造局部序列。

2. `PanoCityPairedOmegaDataset`
   - 读取扁平的 `rgb/`、`depth/` 配对目录。
   - 支持无效深度哨兵值，例如 `65535`。
   - 缺少真实位姿时使用固定步长生成占位位置。

多全景输入形状为 `[B, N, 3, H, W]`。模型先把每个 pano 独立采样成 `S` 个窗口，再合并成 `[B, N*S, 3, window, window]`，同时保留每个 token 所属的 `pano_id`。

当前主要实验配置使用 `pano_sample_mode: single`，即每个训练样本只有一个 ERP 全景。

### 3.2 ERP 到虚拟透视窗口

`PanoWindowSampler` 根据 yaw、pitch 和水平/垂直 FoV 构造 pinhole rays，再映射到 ERP 的 `(u, v)` 坐标，通过 `grid_sample` 得到透视窗口。

当前常用设置为：

- 窗口尺寸：`512 x 512`
- patch size：`16`
- yaw 数量：`8`
- FoV：`75°`
- pitch：通常为 `0°`；PanoCity 稳定性实验使用 `-15°`

ERP 水平方向通过首尾列 padding 实现周期采样，避免直接在经度接缝处越界。

采样器同时生成两类结构化元数据：

- 每个 patch 的元数据：
  - `global_patch_id`
  - ERP 坐标
  - 球面角和三维方向
  - 局部 patch 坐标
  - window/pano ID
  - 接缝区域标记
- 每个虚拟窗口的相机元数据：
  - yaw、pitch、FoV
  - camera encoding
  - 已知旋转矩阵
  - 同一 pano 下为零的虚拟相机平移

### 3.3 VGGT-Omega 与 LUNA 特征融合

原始 Omega 的 24 层 Aggregator 仍保持：

```text
frame attention -> inter-frame attention
```

其中部分层使用全 token 的 global attention，部分层使用只在特殊 token 间交互的 register attention。

LUNA 被插入在每层完整的 frame/inter-frame attention 之后、稀疏特征缓存之前，因此不会改变原始 attention block 的内部实现。

#### LUNA-Patch

LUNA-Patch 根据 `global_patch_id`，把来自不同透视窗口、但落在同一 ERP 全局区域的 patch token 聚合成一个 batch-local patch bank。每个 patch 的修正输入由以下内容组成：

```text
当前 patch 特征
+ 对应全局 patch-bank 的均值特征
+ 球面位置编码
```

修正结果通过可学习标量 `alpha` 以残差形式加入。`alpha` 初始化为 0，因此加载原始 VGGT-Omega 权重时，新增模块初始行为是严格 no-op。

#### LUNA-Camera

LUNA-Camera 把已知的虚拟相机 yaw、pitch、FoV 等编码成残差，注入 camera token。当前默认只在第 23 层使用，使最终 CameraHead 能直接看到虚拟窗口几何信息。

#### Pano-global token

项目在 camera token 与 register tokens 之间增加一个 pano-global token。每个窗口的全景几何编码同时加到 camera token 和 pano-global token，用于提供窗口在完整球面中的位置上下文。

启用该 token 后：

```text
patch_token_start = 1 camera + 1 pano-global + 16 registers = 18
```

原始 Omega 的 DenseHead、CameraHead 和 TextAlignmentHead 均通过动态的 `patch_token_start` 读取 token，因此不需要硬编码修改。

### 3.4 深度监督目标构造

这是当前代码相对初始 guide 最重要的工程扩展之一。

VGGT DenseHead 预测的是虚拟 pinhole 相机坐标系中的 Z-depth，而 ERP 数据的深度可能具有不同含义。当前代码显式支持：

- `range`：沿 ERP 球面射线的径向距离。
- `cubemap_z`：单 cubemap face 的投影 Z-depth。
- `double_cubemap_z`：双 cubemap 融合产生的投影深度近似。

监督目标构造顺序为：

1. 将原始 ERP 深度解码为 radial range。
2. 在 ERP 上先构造有效深度 mask。
3. 将“深度乘有效权重”和“有效权重”一起做双线性窗口采样。
4. 用采样权重归一化深度，防止无效值污染邻近像素。
5. 对原始有效 mask 做最近邻采样。
6. 根据窗口光线与光轴夹角，将 radial range 转换为 pinhole Z-depth。
7. 只在有限、正值且小于 `depth_max_m` 的像素上计算损失。

这避免了把 ERP radial depth 直接当成 pinhole Z-depth 的几何错误，也避免 `inf` 或无效深度经双线性插值扩散。

### 3.5 损失函数与训练参数

总损失为：

```text
L = L_depth + camera_loss_weight * L_camera
```

深度损失在 log-depth 空间计算，支持：

- `log_l1`
- `log_huber`
- `clipped_log_l1`

其中 PanoCity 的 pitch-down 稳定配置使用 `clipped_log_l1`，限制极端深度误差的梯度影响。

预测深度还支持一个正值尺度参数：

- 固定尺度：先通过数据集标定获得，再写入训练配置。
- 可学习尺度：优化 `log(scale)`，取指数后乘到预测深度，保证尺度始终为正。

当前单全景主配置采用以下策略：

- `camera_supervision_mode: none`
- 相机损失权重全部为 0
- 主要训练 `luna_dense`，即 LUNA、pano-global/geometry 参数和 DenseHead
- 部分早期配置使用 `luna_heads` 和可学习深度尺度

因此，当前单全景实际优化目标是 metric depth reconstruction，而不是 CameraHead 位姿回归。

多全景代码仍支持 `pano_relative` 相机监督：先把每个窗口预测的平移恢复为 pano center，同一 pano 的窗口中心取均值，再监督各 pano 相对 anchor pano 的平移，并加入同 pano 窗口中心一致性项。

### 3.6 训练运行与检查点

训练入口为：

```bash
python launch.py --config <config.yaml>
```

当前训练框架支持：

- 单卡或 DDP
- bf16 autocast
- AdamW
- 梯度裁剪
- 按 step 或最长运行时间停止
- CSV loss 日志和曲线
- 高误差窗口的预测/GT 深度调试图
- full checkpoint
- 只保存可训练参数的 `trainable_delta` checkpoint

`trainable_delta` 会记录基础 Omega checkpoint 路径、训练参数、预测深度尺度和当前 step，适合 24 GB GPU 上的轻量实验。

### 3.7 推理、重建与评估

`scripts/reconstruct_pano_omega.py` 会：

1. 从 checkpoint 恢复训练时的采样和模型参数。
2. 对单个 ERP 采样虚拟窗口并预测 Z-depth、置信度和 pose encoding。
3. 过滤超过 ERP radial-depth 上限的窗口角点。
4. 输出窗口深度图和简单的 ERP depth splat。
5. 输出两类预测点云：
   - `pred_known_window_camera_points.ply`：使用采样器已知的窗口几何。
   - `pred_official_camera_points*.ply`：使用 CameraHead 预测的 pose。
6. 输出 GT 的窗口点云和 ERP 点云作为对照。

当前单全景配置没有训练 CameraHead，因此几何评估应优先使用 known-window reconstruction；CameraHead 点云更适合诊断，不应被视为当前主结果。

项目还提供：

- 固定深度尺度标定。
- checkpoint 深度误差评估。
- 原始 Omega 与 pano wrapper 的等价性检查。
- GT 深度语义对比。
- 点云尺度统计。
- Sim3 ICP 对齐和重建可视化对比。

## 4. 与初始 guide 的对照

| 项目 | 初始 guide | 当前实现 |
|---|---|---|
| 核心目标 | 把 LUNA 从 b1 移植到 Omega | 保留该结构，并发展为完整训练和重建管线 |
| 输入 | 单 ERP 或普通多视图 | 单 ERP、固定多 pano、可变 pano 邻域 |
| 全景采样 | 虚拟 pinhole 窗口与元数据 | 已用于 RGB、深度、有效 mask 和多 pano flatten |
| LUNA-Patch | 跨窗口共享全局 patch 特征 | 已实现并参与可选择微调 |
| LUNA-Camera | 注入已知窗口相机信息 | 已实现；单 pano 主配置目前不使用 CameraHead loss |
| pano-global token | 可选结构增强 | 当前训练模型默认启用 |
| 权重兼容 | 新模块零初始化，`strict=False` 加载 | 保留，并增加基础权重 + delta checkpoint 机制 |
| 验证范围 | sampler、adapter、aggregator smoke test | 增加深度语义、相机监督模式和数据邻域测试 |
| 深度训练 | 初始 guide 未给出完整 metric depth 路径 | 已实现 ERP depth 解码、有效采样、Z-depth 转换和稳健损失 |
| 相机监督 | 初始版本未形成完整训练约束 | 已实现 window pose 与 pano-relative；当前单 pano 配置关闭相机损失 |
| 重建输出 | 初始 guide 未包含 | 已支持窗口/ERP 深度、多个点云路径、尺度标定和 Sim3 评估 |

## 5. 当前工作的主要创新点

### 5.1 保留局部透视推理，同时补充全景全局一致性

项目没有直接把 Omega 改造成原生 ERP Transformer，而是继续使用其擅长的 pinhole 局部推理，再通过 LUNA 把不同窗口映射回同一 ERP patch bank。这种设计复用了预训练模型的局部几何能力，同时为重叠区域和经度接缝提供跨窗口信息通道。

### 5.2 几何显式的轻量残差适配

球面方向、全局 patch ID 和虚拟相机参数不是只在数据预处理阶段使用，而是显式进入 token 更新。新增模块采用零初始化残差，兼顾：

- 原始 Omega 权重兼容。
- 初始数值行为稳定。
- 只训练少量新增参数和任务 head。

### 5.3 正确连接 ERP 深度与 VGGT 的 Z-depth 定义

当前代码区分 radial range、cubemap Z 和 pinhole Z，先恢复真实射线距离，再转换到每个虚拟窗口的光轴深度。这是保证 metric supervision 几何正确性的关键，比直接 resize/crop ERP 深度更严格。

### 5.4 无效深度感知的窗口采样

深度和有效权重联合采样、再归一化的做法，解决了无效像素、无穷深度和全景空洞在双线性采样中污染邻域的问题。最近邻有效 mask 又避免了把插值权重误当成真实有效标签。

### 5.5 单全景与多全景相机目标分离

单全景的多个窗口共享同一真实相机中心，不能把它们误当成独立移动相机。项目将训练目标拆成：

- 单全景：当前主配置只做深度监督。
- 多全景：监督 pano-level 相对中心，而不是逐窗口的伪世界位姿。

这种分离避免了用不合理的绝对 UE/world 坐标约束单全景虚拟相机。

### 5.6 训练、标定和重建闭环

当前项目不只输出训练 loss，还提供：

- 深度尺度标定。
- 稳健深度损失。
- 高误差窗口调试。
- known-window 与 predicted-camera 两套重建。
- point-cloud Sim3 对齐。
- 原模型和全景包装器的等价性检查。

这使“模型是否学习到正确深度”和“点云为什么错位”可以被分开诊断。

## 6. 当前实现边界

1. 当前主要配置是单全景 depth-only 训练。虽然 CameraHead 仍可前向输出 pose，但它在 `luna_dense` 配置中被冻结且没有相机损失。
2. 单目/单全景仍存在全局尺度歧义。当前通过固定标定尺度或单独的可学习正尺度处理，并没有从单张 RGB 中消除该理论歧义。
3. PanoCity paired 数据没有真实相机轨迹，代码使用顺序生成的占位位置；因此其多全景相机监督不能等价于真实轨迹监督。
4. 当前输出仍以虚拟窗口为基本预测单元，ERP 深度由窗口结果 splat 得到；没有原生 ERP dense decoder 或显式 seam loss。
5. LUNA-Patch 使用量化的 `global_patch_id` 做均值聚合，属于轻量全景一致性机制，不是连续球面上的精确可见性、遮挡或特征重投影。
6. 当前测试覆盖结构、几何约定和 smoke path，但不能替代完整数据集上的精度、泛化和长时间训练验证。
7. 根 README 中曾加入“单 pano 使用 local-zero 相机监督”的说明，但当前 `docs/pano_training_settings.md` 和实际单 pano 配置已改为 `camera_supervision_mode: none`。以当前配置和训练代码为准。

## 7. 关键代码入口

- 原始设计说明：`README_LUNA_OMEGA.md`
- 全景采样：`vggt_omega/data/pano_sampler.py`
- LUNA 包装模型：`vggt_omega/models/vggt_omega_luna.py`
- Omega 注入位置：`vggt_omega/models/aggregator.py`
- LUNA-Patch：`vggt_omega/models/layers/luna_patch.py`
- LUNA-Camera：`vggt_omega/models/layers/luna_camera.py`
- 训练主流程：`training/train_pano_omega.py`
- 数据集：`training/data/pano_vkitti.py`、`training/data/pano_city_paired.py`
- 当前训练配置：`configs/`
- 重建导出：`scripts/reconstruct_pano_omega.py`
- 尺度标定：`scripts/calibrate_depth_scale.py`
- 深度评估：`scripts/evaluate_depth_checkpoint.py`
- Sim3 对齐：`scripts/align_pointcloud_sim3.py`
- 测试：`tests/`
