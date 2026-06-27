# CNN 基础

## 核心组件

### 卷积层
- 感受野、步长（stride）、填充（padding）
- 输出尺寸：`(W - K + 2P) / S + 1`
- 深度可分离卷积（Depthwise Separable Conv）：MobileNet 的核心，减少参数量

### 批归一化（Batch Normalization）
- 加速训练，缓解梯度消失
- 训练时用 batch 统计，推理时用滑动平均
- 变体：LayerNorm（Transformer）、GroupNorm（小 batch）

### 激活函数

| 函数 | 优点 | 缺点 |
|------|------|------|
| ReLU | 计算简单，无梯度消失 | Dead Neuron |
| LeakyReLU | 解决 Dead Neuron | — |
| GELU | Transformer 常用 | 计算略复杂 |
| Sigmoid/Tanh | 输出有界 | 深层梯度消失 |

### 池化层
- MaxPooling：保留最强特征
- AvgPooling：全局平均池化（GAP）常用于分类头
- AdaptiveAvgPool：输出固定尺寸，与输入无关

---

## 卷积计算 PyTorch 代码示例

```python
import torch
import torch.nn as nn

# 标准 2D 卷积
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
x = torch.randn(1, 3, 224, 224)
out = conv(x)  # shape: (1, 64, 224, 224)

# 深度可分离卷积（Depthwise Separable Conv）
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 深度卷积：每通道独立卷积（groups=in_ch）
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        # 逐点卷积：1×1 融合通道信息
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# 参数量对比：标准 3×3 vs 深度可分离
# 标准：3×3×in_ch×out_ch
# 可分离：3×3×in_ch + 1×1×in_ch×out_ch（约减少 8-9 倍）

# 1×1 卷积：通道变换，不改变空间尺寸
conv1x1 = nn.Conv2d(256, 64, kernel_size=1)  # 降维
```

---

## 感受野计算

感受野（Receptive Field）表示输出特征图上每个点能"看到"的输入区域大小。

**逐层累积公式**：

```
RF_l = RF_{l-1} + (K_l - 1) × ∏_{i=1}^{l-1} stride_i
```

**简化版本（全程 stride=1）**：L 层 3×3 卷积后感受野 = `1 + 2L`

**示例（ResNet 早期层）**：

| 层 | 卷积核 | stride | 累积 stride 积 | 感受野 |
|----|--------|--------|--------------|--------|
| 输入 | — | — | 1 | 1 |
| conv1（7×7）| 7×7 | 2 | 2 | 7 |
| maxpool（3×3）| 3×3 | 2 | 4 | 11 |
| res2a（3×3×2）| 3×3 | 1 | 4 | 27 |
| res3a（3×3×2）| 3×3 | 2 | 8 | 43 |

**有效感受野（Effective RF）**：实际上并非所有像素贡献相等，中心像素的梯度贡献最大，形成高斯分布形状。增大感受野的方式：堆叠更多层、使用空洞卷积（Dilated Conv）。

```python
# 空洞卷积：在不增加参数的情况下扩大感受野
# dilation=2 时，3×3 卷积的感受野等效为 5×5
dilated_conv = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
```

---

## 四种归一化方式对比

以输入形状 `(N, C, H, W)` 为例，N=batch，C=通道，H/W=空间：

| 归一化 | 统计维度 | 公式（均值μ）| 适用场景 |
|--------|----------|-------------|----------|
| **BN**（Batch Norm）| N、H、W（每通道）| `μ_c = mean_{n,h,w}(x_{n,c,h,w})` | 图像分类，batch≥8 |
| **LN**（Layer Norm）| C、H、W（每样本）| `μ_n = mean_{c,h,w}(x_{n,c,h,w})` | Transformer、RNN |
| **GN**（Group Norm）| H、W、组内通道（每样本）| `μ_{n,g} = mean_{c∈g,h,w}(x_{n,c,h,w})` | 小 batch 检测/分割 |
| **IN**（Instance Norm）| H、W（每样本每通道）| `μ_{n,c} = mean_{h,w}(x_{n,c,h,w})` | 图像风格迁移 |

归一化后执行：`y = γ·(x−μ)/σ + β`，其中 γ、β 是可学习的缩放和偏移参数。

```python
# PyTorch 中四种归一化
bn = nn.BatchNorm2d(64)           # BN：batch 维度统计
ln = nn.LayerNorm([64, 32, 32])   # LN：指定归一化维度
gn = nn.GroupNorm(num_groups=8, num_channels=64)  # GN：8 个组
in_ = nn.InstanceNorm2d(64)       # IN：每个实例独立
```

**选择原则**：
- batch ≥ 16 → BN
- Transformer / 序列模型 → LN
- 目标检测/分割（batch 小，如 2~4）→ GN
- 风格迁移 → IN

---

## 激活函数详解

### 数学公式汇总

| 函数 | 公式 | 值域 | 特点 |
|------|------|------|------|
| ReLU | `f(x) = max(0, x)` | [0, +∞) | 简单高效，负数梯度为 0 |
| LeakyReLU | `f(x) = x if x>0 else αx`（α≈0.01）| (−∞, +∞) | 解决 Dead Neuron |
| PReLU | 同 LeakyReLU，α 可学习 | (−∞, +∞) | 自适应负斜率 |
| ELU | `f(x) = x if x>0 else α(e^x−1)` | (−α, +∞) | 负值均值更接近 0 |
| GELU | `f(x) = x · Φ(x)`，Φ 为标准正态 CDF | (−∞, +∞) | BERT/ViT 标配，平滑 |
| SiLU/Swish | `f(x) = x · sigmoid(x)` | (−∞, +∞) | YOLO/EfficientNet 常用 |
| Mish | `f(x) = x · tanh(softplus(x))` | (−∞, +∞) | YOLOv4 使用 |
| Sigmoid | `f(x) = 1/(1+e^{-x})` | (0, 1) | 二分类输出层 |
| Tanh | `f(x) = (e^x−e^{-x})/(e^x+e^{-x})` | (−1, 1) | 零中心，GAN 生成器 |

### 图形特点描述

- **ReLU**：负半轴输出恒为 0（梯度死亡），正半轴为线性，存在 Dead Neuron 问题
- **GELU**：在 x≈0 附近有轻微负值，整体比 ReLU 更平滑，Transformer 中首选
- **Swish/SiLU**：非单调曲线，在 x≈−1.28 处有最小值约 −0.28，YOLO 系列广泛使用

```python
import torch.nn.functional as F

x = torch.randn(4, 64, 32, 32)
out_relu  = F.relu(x)
out_gelu  = F.gelu(x)
out_silu  = F.silu(x)    # Swish
out_mish  = F.mish(x)
```

---

## 1×1 卷积的作用

1×1 卷积（Pointwise Convolution）不改变空间分辨率，但作用多样：

1. **通道升降维**：减少计算量（ResNet Bottleneck 的核心）
   - `256→64→256`，先降维再升维，减少 3×3 卷积的 FLOPs
2. **跨通道特征融合**：将不同通道线性组合，类似全连接
3. **引入非线性**：后接 BN + ReLU 增加表达能力
4. **维度匹配**：残差连接中调整通道数使 shortcut 可相加

```python
# Bottleneck 中 1×1 卷积用法
class Bottleneck(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),   # 1×1 降维
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),  # 3×3 提特征
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),  # 1×1 升维
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False) \
            if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.relu(self.net(x) + self.shortcut(x))
```

---

## Dropout 变体

### 标准 Dropout
随机将神经元输出置零，防止特征共适应（co-adaptation）。训练时按概率 p 丢弃，推理时关闭。

```python
dropout = nn.Dropout(p=0.5)
```

### DropPath（Stochastic Depth）
**整条残差路径**随机丢弃，而非单个神经元。Swin Transformer、DeiT、ConvNeXt 等广泛使用，随层深度线性增大丢弃率。

```python
from timm.models.layers import DropPath

class SwinBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

### DropBlock
对特征图上的**连续空间块**进行丢弃，防止空间位置信息泄漏，比 Dropout 更适合 CNN。

```python
# timm 库提供完整实现
from timm.models.layers import DropBlock2d

# block_size：丢弃块的大小，通常为 5~7
# drop_prob：丢弃概率
drop_block = DropBlock2d(drop_prob=0.1, block_size=7)
```

**对比总结**：

| 变体 | 作用粒度 | 适用场景 | 推荐丢弃率 |
|------|----------|----------|-----------|
| Dropout | 单神经元 | 全连接层、NLP | 0.1~0.5 |
| DropPath | 整个残差分支 | Transformer、残差网络 | 0.0~0.3（随深度线性增大）|
| DropBlock | 空间连续块 | CNN 卷积特征图 | 0.1~0.2 |
