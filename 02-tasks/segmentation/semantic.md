# 语义分割

> 对图像每个像素进行类别预测，不区分同类不同实例。

## 经典架构

### 编码器-解码器

| 模型 | 特点 |
|------|------|
| FCN | 第一个端到端语义分割网络 |
| U-Net | 跳跃连接，医学图像标配 |
| SegNet | 最大池化索引上采样 |
| DeepLabV3+ | ASPP 空洞卷积 + 解码器，工业常用 |
| HRNet | 高分辨率表示，多尺度并行 |

### Transformer-based

| 模型 | 特点 |
|------|------|
| SETR | ViT 骨干做分割 |
| SegFormer | 轻量 MiT 骨干，简洁解码器，强烈推荐 |
| Mask2Former | 统一框架，语义/实例/全景 |
| OneFormer | 单模型三任务 |

## 关键技术

- **空洞卷积（Dilated Conv）**：增大感受野不降低分辨率
- **ASPP**：多个膨胀率并行，捕获多尺度上下文
- **上采样**：双线性插值 vs 转置卷积

---

## U-Net 完整 PyTorch 实现

U-Net 由编码器（下采样路径）和解码器（上采样路径）组成，通过跳跃连接融合浅层细节与深层语义。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    """MaxPool 下采样 + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    """双线性上采样 + 跳跃连接 + DoubleConv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理特征图尺寸不整除的情况
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx//2, dx - dx//2, dy//2, dy - dy//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super().__init__()
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // (2 if bilinear else 1))
        self.up1   = Up(1024, 512 // (2 if bilinear else 1), bilinear)
        self.up2   = Up(512, 256 // (2 if bilinear else 1), bilinear)
        self.up3   = Up(256, 128 // (2 if bilinear else 1), bilinear)
        self.up4   = Up(128, 64, bilinear)
        self.outc  = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)  # [B, n_classes, H, W]
```

**踩坑记录**：医学图像分割中 U-Net 输入尺寸最好是 32 的倍数，避免上采样时 padding 引入误差；跳跃连接建议拼接（concat）而非相加，保留更多低层信息。

---

## DeepLabV3+ ASPP 模块实现

ASPP（Atrous Spatial Pyramid Pooling）通过多个不同膨胀率的空洞卷积并联，捕获多尺度上下文：

```python
class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class ASPPPooling(nn.Sequential):
    """全局平均池化分支"""
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_ch, atrous_rates=(6, 12, 18), out_ch=256):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
            )  # 1×1 卷积分支
        ]
        modules += [ASPPConv(in_ch, out_ch, r) for r in atrous_rates]
        modules.append(ASPPPooling(in_ch, out_ch))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.project(torch.cat([m(x) for m in self.convs], dim=1))
```

---

## SegFormer：轻量级架构

SegFormer 使用**混合 Transformer 编码器（MiT）+ 全 MLP 解码器**，无需位置编码，适用于任意分辨率。

```
MiT 编码器（4 个阶段，逐步降低分辨率）：
  Stage 1: H/4 × W/4，C=64
  Stage 2: H/8 × W/8，C=128
  Stage 3: H/16 × W/16，C=320
  Stage 4: H/32 × W/32，C=512

全 MLP 解码器：
  ↓ 各阶段特征 → 1×1 卷积统一通道 → 双线性上采样到 H/4
  ↓ 拼接 → MLP Fusion → 上采样 → 分类 logits
```

```python
# 使用 transformers 库加载 SegFormer
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")

from PIL import Image
img = Image.open("street.jpg")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
# 上采样到原图大小
logits = outputs.logits  # [1, num_classes, H/4, W/4]
upsampled = torch.nn.functional.interpolate(logits, size=img.size[::-1],
                                             mode='bilinear', align_corners=False)
pred = upsampled.argmax(dim=1)  # [1, H, W]
```

---

## 后处理：CRF（条件随机场）

CRF 可以利用颜色信息精化分割边界，常用 `pydensecrf`：

```python
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_crf(image, prob_map, n_iters=10):
    """
    image:    [H, W, 3]，uint8
    prob_map: [C, H, W]，softmax 后的概率图
    """
    H, W, C_img = image.shape
    n_classes = prob_map.shape[0]
    d = dcrf.DenseCRF2D(W, H, n_classes)

    # 一元势（来自网络输出）
    unary = unary_from_softmax(prob_map)
    d.setUnaryEnergy(unary)

    # 二元势（外观核：颜色相似 + 位置相近）
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

    Q = d.inference(n_iters)
    return np.argmax(Q, axis=0).reshape(H, W)
```

**注意事项**：CRF 推理速度较慢（单张约 100ms），生产环境可用更轻量的后处理替代；对细线、毛发等边界效果明显，对大面积区域提升不大。

---

## mIoU 计算（NumPy 实现）

```python
import numpy as np

def compute_miou(pred, target, num_classes, ignore_index=255):
    """
    pred:   [H, W]，预测类别索引
    target: [H, W]，GT 类别索引
    """
    ious = []
    valid_mask = target != ignore_index
    pred, target = pred[valid_mask], target[valid_mask]

    for cls in range(num_classes):
        pred_cls = pred == cls
        gt_cls   = target == cls
        intersection = (pred_cls & gt_cls).sum()
        union        = (pred_cls | gt_cls).sum()
        if union == 0:
            continue  # 该类在预测和 GT 中都不存在，跳过
        ious.append(intersection / union)

    return np.mean(ious)

# 使用混淆矩阵批量计算（更高效）
def fast_hist(pred, label, n_classes):
    """生成混淆矩阵"""
    mask = (label >= 0) & (label < n_classes)
    return np.bincount(
        n_classes * label[mask].astype(int) + pred[mask],
        minlength=n_classes ** 2
    ).reshape(n_classes, n_classes)

def per_class_iou(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-9)
```

---

## 类别不均衡处理

语义分割中常见背景类像素远多于前景类，常用方案：

```python
# 方案1：加权交叉熵（权重与类别频率成反比）
class_weights = 1.0 / (class_pixel_freq + 1e-9)
class_weights = class_weights / class_weights.sum() * num_classes
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float())

# 方案2：Focal Loss（动态降低易分类样本权重）
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# 方案3：Dice Loss（直接优化 IoU 指标）
def dice_loss(pred, target, smooth=1.0):
    pred = pred.sigmoid()
    intersection = (pred * target).sum(dim=(2, 3))
    return 1 - (2 * intersection + smooth) / (
        pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
    )
```

## 评估指标

- **mIoU**（mean Intersection over Union）：各类 IoU 均值
- **pixel Accuracy**：像素级准确率（不均衡场景不可靠）
