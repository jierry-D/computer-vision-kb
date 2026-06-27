# Transformer 在视觉中的应用

## 核心机制

- **自注意力（Self-Attention）**：每个 token 与所有其他 token 计算关联
- **位置编码**：补充位置信息（绝对/相对）
- **Multi-Head Attention**：多组注意力并行，捕捉不同子空间特征

## 主要模型

### ViT（Vision Transformer）
- 将图像切成固定 patch（16×16），展平后当 token
- 纯 Transformer 编码器
- 缺点：需要大量数据预训练

### Swin Transformer
- 分层结构（类似 CNN 的多尺度）
- 窗口注意力（W-MSA）+ 移位窗口（SW-MSA），降低计算复杂度
- 目前检测/分割最强骨干之一

### DeiT
- 数据高效的 ViT，加入蒸馏 token

### DETR
- 用 Transformer 做端到端目标检测，去掉 NMS
- 延伸：Deformable DETR、DINO（检测）

## 与 CNN 对比

| 维度 | CNN | Transformer |
|------|-----|------------|
| 归纳偏置 | 强（局部性、平移不变）| 弱（需更多数据）|
| 全局感受野 | 需堆叠多层 | 单层即可 |
| 计算复杂度 | O(n) | O(n²)，Swin 改为 O(n) |
| 小数据集 | 更好 | 较差（需预训练）|

---

## Self-Attention 矩阵计算步骤

**输入**：序列 X ∈ R^{N×d}（N 个 token，每个维度为 d）

**第一步：线性投影得到 Q、K、V**

```
Q = X · W_Q    （N × d_k）
K = X · W_K    （N × d_k）
V = X · W_V    （N × d_v）
```

**第二步：计算注意力分数并 Softmax 归一化**

```
Attention(Q, K, V) = softmax( Q·K^T / √d_k ) · V
```

除以 √d_k 防止点积数值过大导致 softmax 梯度消失。

**Multi-Head Attention**：将 d 拆分为 h 个头，每个头独立计算，最后拼接：

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        h = self.n_heads

        # 投影并拆分多头：(B, N, D) → (B, h, N, d_k)
        Q = self.W_Q(x).view(B, N, h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, N, h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, N, h, self.d_k).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,h,N,N)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        # 加权求和
        out = torch.matmul(attn, V)                        # (B,h,N,d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_O(out)
```

---

## 位置编码的两种实现

### 绝对位置编码（APE，Absolute Positional Encoding）

**固定正弦编码（ViT 原始版）**：

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

```python
import numpy as np

def get_sinusoidal_pe(n_pos, d_model):
    pe = np.zeros((n_pos, d_model))
    for pos in range(n_pos):
        for i in range(0, d_model, 2):
            pe[pos, i]   = np.sin(pos / (10000 ** (i / d_model)))
            pe[pos, i+1] = np.cos(pos / (10000 ** (i / d_model)))
    return torch.tensor(pe, dtype=torch.float32)

# 可学习的绝对位置编码（ViT 默认使用）
class LearnableAPE(nn.Module):
    def __init__(self, n_patches, d_model):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed
```

### 相对位置编码（RPE，Relative Positional Encoding）

**Swin Transformer 使用的相对位置偏置（Relative Position Bias）**：

- 不编码绝对位置，而是在注意力分数中加入两个 token 之间的相对位置偏置 B
- 对于窗口内的 M×M 个 token，相对位置范围为 [-(M-1), M-1]，共 (2M-1)² 种
- 偏置表存在可学习的参数矩阵中，双线性插值支持不同窗口大小的迁移

```
Attention = softmax( Q·K^T / √d_k + B ) · V
```

**优点**：对分辨率变化更鲁棒，预训练后迁移到不同尺寸时表现更好

---

## Swin 的 W-MSA 和 SW-MSA 轮流切换详解

Swin Transformer 通过交替使用两种注意力实现局部+跨窗口的全局感受野：

**W-MSA（第 2l 层）**：
- 将特征图 H×W 等分为 `(H/M)×(W/M)` 个不重叠窗口（M=7）
- 每个窗口内 M² 个 token 做 Self-Attention
- 窗口间无信息交流，感受野受限

**SW-MSA（第 2l+1 层）**：
- 将特征图循环移位 `(⌊M/2⌋, ⌊M/2⌋)`（即 (3,3)）个像素
- 重新划分窗口，使新窗口跨越原窗口边界
- 引入 **Attention Mask**：屏蔽同一新窗口中不相邻原窗口的 token 之间的注意力

**循环移位的高效实现**：

```python
import torch

# 循环移位：将特征图向左上偏移 shift_size 个像素
def cyclic_shift(x, shift_size):
    # x: (B, H, W, C)
    return torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

# 对应的逆向移位（用于恢复原始位置）
def reverse_cyclic_shift(x, shift_size):
    return torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
```

通过这种机制，仅使用 O(N) 的计算复杂度就能实现全局感受野。

---

## DETR 的二分匹配（匈牙利算法）

DETR 抛弃了传统目标检测中的 NMS（非极大值抑制），使用**集合预测**方式直接输出 N 个候选框。

**问题**：N 个预测框与 M 个真实框（M≤N）如何配对？

**解决方案：匈牙利算法（二分图最优匹配）**

```
步骤：
1. 构建代价矩阵 C（N×N），其中 C[i][j] 表示第 i 个预测框与第 j 个真实框的匹配代价
   代价 = λ_cls · L_cls(p_i, c_j) + λ_L1 · ||b_i - b_j|| + λ_iou · L_iou(b_i, b_j)

2. 用匈牙利算法求最优一对一匹配（最小化总代价）

3. 匹配后计算最终损失（仅对匹配上的预测框计算分类+回归损失，未匹配的预测为"无目标"类）
```

```python
from scipy.optimize import linear_sum_assignment

def hungarian_matching(cost_matrix):
    """
    cost_matrix: (n_pred, n_gt) numpy array
    返回匹配的预测索引和真实框索引
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind
```

优点：端到端训练，无需 anchor、NMS、预设 anchor 尺寸；缺点：训练收敛慢（需 500 epoch），已被后续 Deformable DETR 改进。

---

## Deformable Attention 原理

Deformable DETR 解决了原版 DETR 训练慢和 Attention Map 稀疏的问题。

**核心思想**：不对所有 token 计算注意力，而是对每个 query **自适应地采样少数关键点**（如 4 个），只在这些采样点处计算注意力。

**公式**：

```
DA(q, p, x) = Σ_{m=1}^{M} W_m · Σ_{k=1}^{K} A_{mqk} · W_m' · x(p + Δp_{mqk})
```

- `p`：query 的参考点（通常为当前特征位置）
- `Δp_{mqk}`：第 m 个注意力头、第 k 个采样点的**偏移量**（可学习）
- `A_{mqk}`：对应的注意力权重（softmax 归一化）
- `x(p + Δp)`：在偏移位置通过双线性插值采样特征值

**优点**：
- 复杂度从 O(HW × HW) 降为 O(HW × K)（K 通常为 4）
- 自然具备多尺度特征融合能力（Multi-Scale Deformable Attention）
- 收敛速度比原版 DETR 快 10 倍以上
