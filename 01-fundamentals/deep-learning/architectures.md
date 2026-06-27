# 主流骨干网络

## 经典架构演进

| 年份 | 网络 | 核心贡献 |
|------|------|---------|
| 2012 | AlexNet | 深度学习 CV 起点，GPU 训练 |
| 2014 | VGG | 统一 3×3 卷积堆叠 |
| 2014 | GoogLeNet/Inception | 多尺度并行卷积（Inception 模块）|
| 2015 | ResNet | 残差连接，解决退化问题 |
| 2017 | DenseNet | 密集连接，特征复用 |
| 2017 | MobileNetV1 | 深度可分离卷积，轻量化 |
| 2018 | EfficientNet | NAS 复合缩放 |
| 2020 | RegNet | 可规律化设计空间 |
| 2021 | ViT | 纯 Transformer 做分类 |
| 2021 | Swin Transformer | 分层 + 窗口注意力，通用视觉骨干 |
| 2022 | ConvNeXt | 现代化 CNN，对标 Swin |

## 选型建议

| 场景 | 推荐 |
|------|------|
| 移动端部署 | MobileNetV3 / EfficientNet-Lite |
| 高精度桌面端 | Swin-L / ConvNeXt-L |
| 检测/分割骨干 | ResNet50 / Swin-T / ConvNeXt-T |
| 快速实验 | ResNet50 |

---

## ResNet 残差块代码实现

核心思想：`输出 = F(x) + x`，让网络学习"残差"而非"映射"，解决深层网络退化（degradation）问题。

```python
import torch.nn as nn
import torch.nn.functional as F

# BasicBlock（ResNet-18 / ResNet-34）
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample  # shortcut 维度不一致时用 1×1 卷积对齐

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)   # 残差相加后激活


# Bottleneck（ResNet-50 / 101 / 152）：1×1→3×3→1×1，减少计算量
class Bottleneck(nn.Module):
    expansion = 4  # 输出通道是 mid_ch 的 4 倍

    def __init__(self, in_ch, mid_ch, stride=1, downsample=None):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)
```

---

## MobileNetV2 倒残差块（Inverted Residual）

MobileNetV2 的关键创新：与 ResNet Bottleneck 方向相反，先**升维**再做深度卷积再**降维**。

**结构**（stride=1 时有 shortcut）：

```
输入(in_ch) → 1×1 升维(×t) → 3×3 DW卷积 → 1×1 降维(out_ch) → (+输入) → 输出
```

- t（expand ratio）：膨胀倍率，通常为 6
- 低维输入/输出 + 高维内部处理：在信息丰富的高维空间进行特征提取
- **Linear Bottleneck**：降维后不加激活函数（避免信息损失）

```python
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio=6):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden),
                       nn.ReLU6(inplace=True)]
        layers += [
            # 深度卷积：groups=hidden，每通道独立卷积
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            # 逐点卷积：降维，不加激活（Linear Bottleneck）
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)
```

---

## EfficientNet 复合缩放公式

EfficientNet 通过**统一缩放**网络宽度、深度、分辨率来提升效率，避免单独调整某一维度的次优问题：

```
depth:      d = α^φ
width:      w = β^φ
resolution: r = γ^φ

约束：α · β² · γ² ≈ 2  （保证 FLOPS ≈ 2^φ 倍增长）
      α ≥ 1, β ≥ 1, γ ≥ 1
```

- φ 是可调节的复合系数（用户控制资源预算）
- α、β、γ 通过小网格搜索在 B0 基础上确定：α=1.2，β=1.1，γ=1.15
- B0~B7：φ 从 0 增至 7，参数量从 5.3M 增至 66M

| 模型 | 输入分辨率 | 参数量 | Top-1（ImageNet） |
|------|-----------|--------|-------------------|
| EfficientNet-B0 | 224 | 5.3M | 77.1% |
| EfficientNet-B3 | 300 | 12M | 81.6% |
| EfficientNet-B4 | 380 | 19M | 82.9% |
| EfficientNet-B7 | 600 | 66M | 84.3% |

---

## Swin Transformer Window Partition 详解

Swin 的核心思想是**在局部窗口内计算自注意力**，将复杂度从 O(N²) 降至 O(N)。

**W-MSA（Window Multi-head Self-Attention）**：
1. 将 H×W 的特征图均匀划分为 `(H/M) × (W/M)` 个不重叠窗口，每个窗口大小 M×M（默认 M=7）
2. 每个窗口内部独立计算 Self-Attention（共 `HW/M²` 个并行的局部注意力）
3. 窗口之间**无信息交流** → 感受野受限于窗口大小

**SW-MSA（Shifted Window Multi-head Self-Attention）**：
1. 将整个特征图偏移 `(M/2, M/2)` 个像素
2. 重新划分窗口后计算注意力
3. 不同位置的窗口可以跨越原始窗口边界 → 实现跨窗口信息交互

**交替使用机制**：

```
第 2l 层：   W-MSA（固定窗口）  → 学习局部特征
第 2l+1 层： SW-MSA（移位窗口）→ 学习跨窗口关系
```

**高效实现**：移位后通过**循环位移 + Attention Mask**（屏蔽不相邻区域的注意力）来避免额外 padding 开销，保持计算规整。

**相对位置偏置**：Swin 使用相对位置偏置（Relative Position Bias）而非绝对位置编码，对不同尺寸的输入有更好的泛化性。

---

## 各网络参数量与 FLOPs 对比

| 网络 | 参数量 | FLOPs（224²）| Top-1（IN1K）| 特点 |
|------|--------|-------------|-------------|------|
| ResNet-50 | 25M | 4.1G | 76.1% | 经典基准 |
| ResNet-101 | 45M | 7.9G | 77.4% | 更深 |
| MobileNetV2 | 3.4M | 300M | 72.0% | 移动端 |
| MobileNetV3-L | 5.4M | 219M | 75.2% | 搜索优化 |
| EfficientNet-B0 | 5.3M | 0.4G | 77.1% | NAS 设计 |
| EfficientNet-B4 | 19M | 4.2G | 82.9% | 高精度 |
| ViT-B/16 | 86M | 17.6G | 81.8% | 纯 Transformer |
| Swin-T | 28M | 4.5G | 81.3% | 轻量 Swin |
| Swin-B | 88M | 15.4G | 83.5% | 高精度 Swin |
| ConvNeXt-T | 28M | 4.5G | 82.1% | 现代 CNN |
| ConvNeXt-B | 89M | 15.4G | 83.8% | 大模型 CNN |

注：FLOPs 为单样本推理浮点运算次数。

---

## 如何选择骨干网络（决策树）

```
部署目标
    ├── 移动端 / 嵌入式
    │     └── MobileNetV3-Large / EfficientNet-Lite / GhostNet
    └── GPU 服务器
          ├── 下游任务：图像分类
          │     ├── 数据 < 10k → ResNet50（预训练微调）
          │     ├── 数据充足，精度优先 → EfficientNet-B4/B5 / ConvNeXt-B
          │     └── 需要快速迭代 → ResNet50
          ├── 下游任务：目标检测
          │     ├── 精度优先 → Swin-T/S（接 FPN/PAN）
          │     ├── 速度优先 → ResNet50 / ConvNeXt-T
          │     └── 实时检测 → CSPDarknet（YOLO 系列）
          └── 下游任务：语义/实例分割
                ├── 精度优先 → Swin-B / ConvNeXt-L（接 Mask2Former）
                └── 速度优先 → ResNet50 / ConvNeXt-T（接 DeepLabV3+）
```

**通用建议**：

| 优先级 | 推荐 | 理由 |
|--------|------|------|
| 快速 baseline | ResNet50 | 生态成熟，调试方便 |
| 精度与效率平衡 | Swin-T / ConvNeXt-T | 参数量接近 ResNet50 但更强 |
| 极致轻量 | MobileNetV3-Large | FLOPs < 300M，适合边缘设备 |
| 极致精度 | Swin-L / ConvNeXt-XL（预训练）| 配合大规模预训练数据 |
