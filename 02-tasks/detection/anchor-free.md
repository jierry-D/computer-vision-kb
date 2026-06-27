# Anchor-Free 目标检测

## 核心思想

直接预测目标中心点或关键点，不依赖预设 anchor，设计更简洁。

## 主流模型

### 基于关键点

| 模型 | 核心思想 |
|------|---------|
| CornerNet | 预测左上/右下角点对 |
| CenterNet | 预测中心点 + 宽高偏移 |
| ExtremeNet | 预测极值点 |

### 基于密集预测

| 模型 | 核心思想 |
|------|---------|
| FCOS | 每个像素点预测到框四边距离 |
| ATSS | 自适应正样本分配策略 |
| GFL | 联合分类和定位质量分支 |

### Transformer-based

| 模型 | 特点 |
|------|------|
| DETR | 端到端，无 NMS，二分匹配 |
| Deformable DETR | 可变形注意力，收敛更快 |
| DINO（检测） | 目前最强 Transformer 检测器之一 |
| RT-DETR | 实时 DETR，百度出品 |

## Anchor-Free vs Anchor-Based

| 维度 | Anchor-Based | Anchor-Free |
|------|-------------|------------|
| 设计复杂度 | 需手动设计 anchor | 更简洁 |
| 超参敏感 | anchor 尺寸敏感 | 较低 |
| 小目标 | FPN 配合较好 | 需专门设计 |
| 趋势 | 逐渐被取代 | 主流方向 |
