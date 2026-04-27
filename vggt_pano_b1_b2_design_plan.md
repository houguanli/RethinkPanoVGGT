# VGGT 全景适配（b1 / b2）精简设计说明

## 核心思路

将单张全景图 \( \mathcal{P} \) 切分为多个重叠的 pinhole 视图 \( \{I_i\} \)，并利用其已知的几何关系（共享光心、仅旋转变化）对 VGGT 进行约束与改造。

整体分两步：

- **b1：只改训练（loss / LoRA），不改或少改结构**
- **b2：引入全景结构（global camera token + 几何编码）**

---

## b1：基于损失的最小适配

### 输入建模

给定 pano：

\[
\mathcal{P} \rightarrow \{I_1, I_2, ..., I_N\}
\]

每个 view 有已知参数：

\[
(\theta_i, \phi_i, fov_i)
\]

---

### 几何约束

#### 1. 同光心约束（translation）

同一 pano 内：

\[
t_i \approx t_j
\]

实现：

\[
L_{zero\_t} = \frac{1}{N} \sum_i \left\| t_i - \bar{t} \right\|_1
\]

---

#### 2. 绝对旋转约束

\[
L_{abs\_rot} = \min \left( \|q_i - q_i^{gt}\|,\ \|q_i + q_i^{gt}\| \right)
\]

---

#### 3. 相对旋转约束（核心）

\[
R_{ij}^{pred} = R_i^{-1} R_j
\quad,\quad
R_{ij}^{gt} = R_i^{gt^{-1}} R_j^{gt}
\]

\[
L_{rel\_rot} = d(R_{ij}^{pred}, R_{ij}^{gt})
\]

---

### 总损失

\[
L = L_{orig}
+ \lambda_1 L_{zero\_t}
+ \lambda_2 L_{abs\_rot}
+ \lambda_3 L_{rel\_rot}
\]

---

### 伪代码

```python
pred = model(images)

loss = original_loss(pred, batch)

if batch.is_pano:
    loss += lambda_zero_t * zero_translation_loss(pred)
    loss += lambda_abs_rot * abs_rotation_loss(pred, gt)
    loss += lambda_rel_rot * relative_rotation_loss(pred, gt, pairs)