# Pano-VGGT + LUNA 技术路线说明

## 1. 背景与目标

当前任务是将全景图（panorama / equirectangular image）重建能力较自然地接入 VGGT 框架。已有实验表明，将全景图投影成多个局部 pinhole-like windows 后，VGGT 本身已经可以完成较稳定的初步重建，但在细节、跨窗口一致性、局部畸变纠偏和相机预测稳定性上仍存在问题。

因此，核心目标不是重新训练一个全景重建模型，而是在尽可能复用 VGGT 预训练权重的前提下，引入一个轻量增量网络 LUNA，使一张完整 pano 可以被组织成一个统一的 VGGT-compatible multi-view token graph，并在此基础上完成：

1. 多个 virtual pinhole windows 的联合 attention；
2. 同一张 pano 中跨 window 的 patch / feature / geometry 共享；
3. 相机 token 的 pano-aware 纠偏；
4. 重建细节和畸变残差修正；
5. 尽可能保持 VGGT 原始 local pinhole reasoning 能力。

一句话总结：

> VGGT 负责已有的 local pinhole geometry prior；LUNA 负责 pano-specific global coupling 和 residual correction。

---

## 2. 当前两个核心痛点

### 2.1 跨 window 的 pano patch 共享问题

如果直接将一张 pano 切成多个 pinhole-like windows，然后作为多视角输入送入 VGGT，虽然输入形式和 VGGT 预训练分布接近，但模型会把这些 windows 视作普通独立 views。

这会导致一个问题：

> 来自同一张 pano、甚至来自同一球面区域的 patch，在不同 window 的 attention 过程中被当作独立 token 处理，缺少全局身份和共享约束。

因此需要建立一种机制，使每个 patch 同时具备：

- local identity：它在当前 virtual window 中的位置；
- global identity：它在原始 pano / 球面上的位置。

示例：

```python
token_meta = {
    "pano_id": 0,
    "window_id": 5,
    "local_patch_xy": (12, 18),
    "global_uv": (u, v),
    "sphere_dir": (dx, dy, dz),
    "global_patch_id": 18372,
}
```

这样模型可以知道：

```text
window A 的 patch i 和 window B 的 patch j 是否来自同一个 pano 区域；
两个 patch 在球面上是否相邻；
两个 window 是否在 pano seam 附近产生循环连续关系。
```

---

### 2.2 投影畸变与细节纠偏问题

通过 pinhole projection 从 pano 中采样局部 windows，可以显著降低直接在 equirectangular 平面上裁剪带来的畸变。已有实验也显示，VGGT 对这种输入能基本完成重建。

因此，当前不需要把“去畸变”作为主任务重新建模，而应把它视为一个 residual correction 问题：

```text
VGGT 输出基础重建结果
LUNA 学习 pano-specific 残差纠偏
```

LUNA 不应取代 VGGT 主干，而应作为轻量 adapter 插入 attention block 或 block 之间，用于修正：

- 局部投影畸变；
- 跨 window feature 不一致；
- seam 断裂；
- virtual camera 之间的相机预测误差；
- pano-derived windows 被误认为普通独立相机阵列的问题。

---

## 3. 总体技术路线

推荐的整体结构如下：

```text
Pano Image
   ↓
Spherical Virtual Camera Sampler
   ↓
N 个 pinhole-like windows
   ↓
VGGT Patch Embed
   ↓
VGGT Local Attention / Aggregator Blocks
   ↓
LUNA-Patch Adapter
   ↓
LUNA-Camera Adapter
   ↓
VGGT Heads
   ↓
Camera / Depth / Point / Reconstruction Outputs
```

更具体地说：

```text
一张 pano
   ↓
生成多个 VGGT-compatible virtual pinhole windows
   ↓
所有 windows 的 tokens 同时进入 VGGT attention
   ↓
通过 global pano patch identity 建立跨 window 共享关系
   ↓
LUNA 在 attention 过程中进行轻量 residual correction
   ↓
完成 pano-aware reconstruction 和 camera prediction
```

重点不是把整张 pano 直接 flatten 成一个巨大 token sequence，而是：

> 将整张 pano 转换成一组相互耦合的 VGGT views，使它们在同一个 attention graph 中联合建模。

---

## 4. Pano 到 Virtual Pinhole Windows

### 4.1 输入形式

原始 pano 通常是 equirectangular 格式：

```text
Pano: [H_pano, W_pano, 3]
```

每个像素可以映射到球面方向：

```text
(u, v) → (theta, phi) → sphere_dir = (x, y, z)
```

其中：

- `theta` 表示水平 yaw；
- `phi` 表示垂直 pitch；
- `sphere_dir` 是单位球面方向向量。

---

### 4.2 Virtual Camera Sampler

在球面上选择若干个中心方向，为每个方向构造一个虚拟 pinhole camera：

```python
virtual_camera = {
    "window_id": i,
    "yaw": yaw_i,
    "pitch": pitch_i,
    "fov_x": fov_x,
    "fov_y": fov_y,
    "center_ray": center_ray,
    "right_ray": right_ray,
    "up_ray": up_ray,
}
```

每个 virtual camera 从 pano 中采样得到一个局部 perspective window：

```text
Pano → VirtualCamera_i → Window_i: [3, H, W]
```

推荐初始参数：

```yaml
window_size: 518
patch_size: 14
fov_x: 70-80
fov_y: 70-80
overlap: 0.25-0.50
```

原因：

- FOV 太大：局部 pinhole 畸变增强，破坏 VGGT prior；
- FOV 太小：window 数过多，显存和全局连接压力增大；
- 70° 到 80° 是比较稳的初始区间。

---

## 5. Token Metadata 设计

每个 window patch token 都需要记录其局部和全局信息。

推荐数据结构：

```python
@dataclass
class PanoTokenMeta:
    pano_id: int
    window_id: int

    # patch 在当前 window 内的位置
    local_patch_x: int
    local_patch_y: int

    # patch 对应的 pano 平面坐标
    pano_u: float
    pano_v: float

    # patch 对应的球面坐标
    theta: float
    phi: float

    # 单位球面方向
    sphere_dir: tuple[float, float, float]

    # 全局 patch id，可由 pano_u / pano_v 离散化得到
    global_patch_id: int

    # 是否在 pano seam 附近
    is_seam_region: bool = False
```

global patch id 可以通过对 pano 的球面坐标或 uv 坐标离散化得到：

```python
global_x = floor(pano_u / patch_stride_u)
global_y = floor(pano_v / patch_stride_v)
global_patch_id = global_y * num_global_x + global_x
```

注意：

- pano 的水平边界需要 cyclic 处理；
- `u = 0` 和 `u = W - 1` 应被视为相邻；
- 高纬区域可以考虑使用 sphere_dir KNN，而不是仅依赖 uv 网格邻近。

---

## 6. LUNA 的定位

LUNA 是轻量增量网络，不是主重建网络。

建议定位：

```text
VGGT backbone:
  保留原始 local pinhole attention 和几何建模能力

LUNA-Patch:
  建立 pano patch 的跨 window feature sharing

LUNA-Camera:
  将已知 virtual camera prior 注入 camera token

LUNA-Residual:
  对局部畸变、seam、细节重建进行轻量纠偏
```

基本形式：

```text
z_out = VGGT_Block(z_in) + alpha * LUNA(z_mid, pano_meta)
```

其中：

```text
alpha 初始值应接近 0，最好是 learnable scale。
```

这样初始状态接近原始 VGGT，能最大限度避免破坏预训练权重。

---

## 7. LUNA-Patch Adapter

### 7.1 目标

解决：

> 不同 virtual windows 中来自相同或相邻 pano 区域的 patch tokens 缺少共享和一致性的问题。

---

### 7.2 最小实现：Global Patch Bank Pooling

对所有 window tokens，根据 `global_patch_id` 做聚合：

```python
global_feature[g] = mean({z_i | global_patch_id(i) == g})
```

然后将 global feature gather 回每个 token：

```python
z_i_corr = MLP(concat(z_i, global_feature[global_patch_id_i], sphere_encoding_i))
z_i_out = z_i + alpha * z_i_corr
```

结构：

```text
Window Tokens
   ↓
scatter mean by global_patch_id
   ↓
Global Pano Feature Bank
   ↓
gather back to each token
   ↓
MLP residual correction
   ↓
Updated Window Tokens
```

优点：

- 实现简单；
- 显存压力小；
- 适合作为 MVP；
- 不破坏 VGGT 主 attention 结构。

---

### 7.3 进阶实现：Sphere-neighbor Sharing

实际情况下，不同 window 中的 patch 不一定刚好对应同一个 `global_patch_id`。因此可以基于球面方向找邻近 token：

```python
neighbors_i = KNN(sphere_dir_i, global_sphere_bank, k=K)
```

然后做局部 cross-attention：

```python
z_i_corr = CrossAttn(
    query=z_i,
    key=neighbor_features,
    value=neighbor_features,
)
z_i_out = z_i + alpha * z_i_corr
```

推荐先使用小 K：

```yaml
knn_k: 4-16
```

该方法可以处理：

- overlap window 中的近似重叠区域；
- seam 附近的 cyclic continuity；
- 球面邻域连续性；
- patch id 离散误差。

---

## 8. LUNA-Camera Adapter

### 8.1 核心观点

由于 virtual pinhole windows 是从 pano 中按已知 yaw / pitch / fov 采样出来的，因此它们之间的相对旋转关系本身是已知的。

因此，不应让 VGGT 从零开始预测这些 virtual camera 的关系，而应使用：

```text
KnownVirtualCamera_i + PredictedResidual_i
```

即：

```text
Camera_i = KnownVirtualCamera_i ∘ ΔCamera_i
```

其中：

- `KnownVirtualCamera_i` 来自 pano sampler；
- `ΔCamera_i` 是 VGGT / LUNA 预测的 residual correction。

---

### 8.2 Camera Metadata

推荐输入给 LUNA-Camera 的 metadata：

```python
camera_meta = {
    "yaw": yaw,
    "pitch": pitch,
    "fov_x": fov_x,
    "fov_y": fov_y,
    "center_ray": center_ray,
    "right_ray": right_ray,
    "up_ray": up_ray,
    "pano_id": pano_id,
    "window_id": window_id,
}
```

通过小 MLP 编码后注入 camera token：

```python
camera_token = camera_token + alpha_cam * MLP(camera_meta)
```

也可以加入 pano global context：

```python
camera_token = camera_token + CrossAttn(camera_token, pano_global_context)
```

---

### 8.3 相机预测输出建议

推荐将 camera head 改为 residual prediction：

```text
R_pred_i = R_known_i · ΔR_i
t_pred_i = t_known_i + Δt_i
```

对于单张 pano 派生出的 virtual windows，旋转 prior 更可靠；平移是否有意义需要根据任务设定决定。

如果这些 windows 来自同一个 pano capture，本质上它们对应同一个相机中心，仅方向不同。因此可以设置：

```text
t_known_i = shared_pano_center
```

然后只允许网络预测较小的 residual，或直接约束所有 virtual windows 的 camera centers 一致。

---

## 9. Position Encoding 策略

为了尽可能复用 VGGT 预训练权重，不建议一开始大改 VGGT 的 RoPE / positional encoding。

推荐保留两套位置系统：

```text
local position:
  继续使用 VGGT 原始 window 内部 patch position / RoPE

global spherical position:
  只输入给 LUNA adapter
```

也就是：

```text
VGGT attention:
  看到的仍然是普通 pinhole window layout

LUNA:
  额外知道 token 在 pano 球面上的 global position
```

这样能最大限度保留 VGGT 原始权重的有效性。

后续如果 LUNA 方案验证有效，再考虑：

- cyclic x RoPE；
- spherical RoPE；
- geodesic positional encoding；
- pano-specific global attention bias。

---

## 10. Loss 设计

### 10.1 基础重建损失

保留原 VGGT 的重建相关损失：

```text
L_depth
L_point
L_camera
L_normal
L_confidence
```

具体使用哪些取决于数据是否有 GT depth / camera pose / point map。

---

### 10.2 Cross-window Feature Consistency

对于来自相同或相邻 pano 区域的 tokens：

```text
L_feat = || z_i - z_j ||
```

或：

```text
L_feat = 1 - cos(z_i, z_j)
```

其中 `i` 和 `j` 满足：

```text
global_patch_id_i == global_patch_id_j
```

或：

```text
dist_sphere(sphere_dir_i, sphere_dir_j) < threshold
```

---

### 10.3 Cross-window Geometry Consistency

如果两个 tokens 对应相近的球面方向，则它们预测出的 3D 点应保持一致：

```text
L_point_consistency = || X_i - X_j ||
```

该 loss 比 feature consistency 更贴近最终任务。

---

### 10.4 Camera Prior / Residual Loss

对于 virtual windows，已知相对旋转：

```text
R_known_ij = R_known_j · R_known_i^{-1}
```

如果模型预测 camera residual，则可约束 residual 不要过大：

```text
L_camera_residual = || ΔR_i || + λ || Δt_i ||
```

或者约束预测相对旋转接近已知相对旋转：

```text
L_relative_R = dist(R_pred_j · R_pred_i^{-1}, R_known_ij)
```

---

### 10.5 Seam Consistency

pano 的左右边界是循环连续的：

```text
u = 0 和 u = W - 1 相邻
```

因此需要对 seam 附近 tokens 施加一致性约束：

```text
L_seam_feat
L_seam_point
L_seam_depth
```

用于防止 pano 重建在水平接缝处断裂。

---

### 10.6 总 Loss

推荐初始形式：

```text
L_total =
    L_vggt_base
  + λ_feat  * L_feat_consistency
  + λ_point * L_point_consistency
  + λ_cam   * L_camera_residual
  + λ_seam  * L_seam_consistency
```

初始建议：

```yaml
lambda_feat: 0.05
lambda_point: 0.1
lambda_cam: 0.1
lambda_seam: 0.1
```

这些值需要根据实际 loss scale 调整。

---

## 11. 训练策略

### Stage 1：冻结 VGGT，只训练 LUNA

目标：

> 验证轻量 adapter 是否能在不破坏 VGGT 的前提下修正 pano-specific 问题。

冻结：

```text
VGGT patch embed
VGGT aggregator
VGGT heads
```

训练：

```text
LUNA-Patch
LUNA-Camera
adapter scale alpha
```

建议：

```yaml
lr_luna: 1e-4 ~ 1e-3
lr_vggt: 0
epochs: 5-20
```

---

### Stage 2：解冻 VGGT 后几层

如果 Stage 1 有明显收益，再解冻：

```text
last K aggregator blocks
camera head
depth / point head 的后部
```

建议：

```yaml
lr_vggt: 1e-5
lr_luna: 1e-4
K: 2-4
```

---

### Stage 3：联合微调

最后进行低学习率联合训练：

```yaml
lr_backbone: 1e-6 ~ 1e-5
lr_heads: 1e-5 ~ 1e-4
lr_luna: 1e-4
```

注意：

- 不建议一开始全量 finetune；
- 否则容易破坏 VGGT 原本的几何 prior；
- LUNA 应该先证明自己能做 residual correction，再让 VGGT 主干轻微适配。

---

## 12. 推荐 MVP 实现顺序

### Step 1：实现 pano virtual window sampler

输入：

```text
pano image
yaw / pitch grid
fov
window_size
patch_size
```

输出：

```text
windows: [S, 3, H, W]
camera_meta: [S, ...]
token_meta: [S, H/patch, W/patch, ...]
```

---

### Step 2：将 windows 作为多 view 输入 VGGT

形式：

```text
[B, S, 3, H, W]
```

其中：

```text
B: batch size
S: 一张 pano 切出的 virtual windows 数量
```

---

### Step 3：保留 VGGT 原始 local position / RoPE

不要先改 VGGT RoPE。

先让每个 virtual window 看起来仍然是普通 pinhole image。

---

### Step 4：插入 LUNA-Patch Adapter

最小版本：

```text
same global_patch_id mean pooling
MLP residual correction
learnable alpha
```

插入位置：

```text
每 N 个 VGGT blocks 后插入一次
```

或者：

```text
只在后半部分 blocks 插入
```

推荐先从后半部分插入，降低对底层视觉特征的扰动。

---

### Step 5：插入 LUNA-Camera Adapter

将 known virtual camera meta 编码后加到 camera token：

```text
camera_token += alpha_cam * MLP(camera_meta)
```

同时让 camera head 预测 residual：

```text
Camera = KnownCamera ∘ DeltaCamera
```

---

### Step 6：加入一致性 loss

最先加入：

```text
L_point_consistency
L_camera_residual
L_seam_consistency
```

feature consistency 可以稍后加入，因为它可能过度限制 feature 表达。

---

### Step 7：冻结 VGGT 训练 LUNA

先验证：

```text
是否减少 seam error？
是否改善 window overlap 区域？
是否提升 camera prediction 稳定性？
是否减少细节局部畸变？
```

---

## 13. 代码结构建议

推荐新增模块：

```text
vggt/
  models/
    vggt_luna.py
    luna_adapter.py

  layers/
    luna_patch.py
    luna_camera.py
    pano_position.py

  data/
    pano_sampler.py
    pano_token_meta.py

  training/
    pano_loss.py
    luna_loss.py
```

建议职责：

```text
pano_sampler.py:
  pano → virtual pinhole windows

pano_token_meta.py:
  生成每个 token 的 global_patch_id / sphere_dir / seam flag

luna_patch.py:
  global patch bank pooling / sphere-neighbor sharing

luna_camera.py:
  camera metadata injection / residual camera prediction

pano_loss.py:
  cross-window feature / geometry / seam consistency

vggt_luna.py:
  封装 VGGT + LUNA 的整体 forward
```

---

## 14. 关键实现伪代码

### 14.1 Forward Skeleton

```python
def forward_pano(pano_image):
    # 1. pano -> virtual windows
    windows, camera_meta, token_meta = pano_sampler(pano_image)

    # 2. VGGT patch embedding
    tokens = vggt.patch_embed(windows)

    # 3. 原始 VGGT aggregator + LUNA adapter
    for block_id, block in enumerate(vggt.aggregator.blocks):
        tokens = block(tokens)

        if block_id in luna_insert_layers:
            tokens = luna_patch_adapter(tokens, token_meta)

        if block_id in luna_camera_layers:
            tokens = luna_camera_adapter(tokens, camera_meta)

    # 4. VGGT heads
    outputs = vggt.heads(tokens)

    # 5. camera residual compose
    outputs["camera"] = compose_known_camera_and_residual(
        camera_meta,
        outputs["camera_delta"],
    )

    return outputs
```

---

### 14.2 LUNA-Patch Minimal Version

```python
class LunaPatchAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + sphere_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, token_meta):
        # tokens: [B, S, N, C]
        # global_ids: [B, S, N]
        global_ids = token_meta["global_patch_id"]
        sphere_enc = token_meta["sphere_encoding"]

        global_bank = scatter_mean(tokens, global_ids)

        global_feat = gather(global_bank, global_ids)

        corr = self.mlp(torch.cat([tokens, global_feat, sphere_enc], dim=-1))

        return tokens + self.alpha * corr
```

---

### 14.3 LUNA-Camera Minimal Version

```python
class LunaCameraAdapter(nn.Module):
    def __init__(self, dim, camera_meta_dim):
        super().__init__()
        self.camera_mlp = nn.Sequential(
            nn.Linear(camera_meta_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, camera_tokens, camera_meta):
        cam_corr = self.camera_mlp(camera_meta)
        return camera_tokens + self.alpha * cam_corr
```

---

## 15. 需要避免的实现陷阱

### 15.1 不要一开始替换 VGGT RoPE

替换 RoPE 会直接改变 VGGT 的 positional prior，训练风险较高。

建议先：

```text
保留 local RoPE；
global spherical position 只给 LUNA 使用。
```

---

### 15.2 不要把 LUNA 做得太重

如果 LUNA 过重，它会变成另一个 backbone，失去“增量适配”的意义。

建议：

```text
adapter dim 不超过 VGGT dim；
层数控制在 1-2 层 MLP 或小 cross-attention；
alpha 初始为 0。
```

---

### 15.3 不要强制所有重叠 tokens 完全相等

不同 windows 中同一球面区域可能因为：

- 投影角度不同；
- 采样误差不同；
- patch 边界不同；
- 局部上下文不同；

导致 feature 不应完全一致。

因此建议先使用 soft sharing，而不是 hard sharing。

---

### 15.4 不要让 camera head 从零预测 virtual camera

virtual camera 的 yaw / pitch / fov 是已知的，应该作为 prior 注入。

否则模型需要重新学习一个本来已知的几何关系，增加不必要的不稳定性。

---

## 16. 推荐实验对照

为了证明 LUNA 的有效性，建议至少设置以下 ablation：

```text
A. VGGT + pinhole windows
B. VGGT + pinhole windows + LUNA-Patch
C. VGGT + pinhole windows + LUNA-Camera
D. VGGT + pinhole windows + LUNA-Patch + LUNA-Camera
E. D + seam consistency
F. D + sphere-neighbor sharing
```

观察指标：

```text
1. camera prediction error
2. point / depth reconstruction error
3. overlap region consistency error
4. seam region consistency error
5. qualitative 3D reconstruction quality
6. high-frequency detail distortion
```

---

## 17. 推荐当前优先级

当前最值得优先实现的是：

```text
1. pano virtual window sampler
2. token_meta / global_patch_id / sphere_dir 记录
3. all-window joint VGGT input
4. LUNA-Patch minimal adapter
5. LUNA-Camera metadata injection
6. camera residual prediction
7. point / seam consistency loss
```

不要过早做：

```text
1. 完全替换 RoPE
2. 直接整张 pano flatten attention
3. 大规模 global full attention
4. 从零训练 pano backbone
5. 过重的 LUNA backbone
```

---

## 18. 最终技术定义

可以将该方法定义为：

> **LUNA: Lightweight Unified pano-aware Network Adapter for VGGT**

其核心是：

```text
local pinhole compatibility + global spherical consistency
```

其中：

- local pinhole compatibility：通过 virtual pinhole windows 保持 VGGT 输入分布；
- global spherical consistency：通过 global patch identity / sphere-neighbor sharing / camera prior / seam consistency 建立 pano 全局约束；
- lightweight adapter：通过残差式 LUNA 模块微调 attention 过程，尽量保留 VGGT 预训练权重。

最终目标：

> 让 VGGT 仍然像处理普通多视角 pinhole images 一样处理输入，但模型内部通过 LUNA 知道这些 views 实际上来自同一张连续的全景球面图。
