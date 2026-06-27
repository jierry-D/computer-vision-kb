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

---

## FCOS：Center-ness 分支详解

FCOS 在每个特征图位置 (x, y) 预测：
- **4 个距离**：到目标框四条边的距离 (l, r, t, b)
- **类别分类分数**
- **Center-ness 分数**（抑制偏离中心的低质量预测框）

### Center-ness 计算

```python
import torch

def compute_centerness(lrtb):
    """
    lrtb: [N, 4]，分别为 left, right, top, bottom 距离（正值）
    返回 centerness target: [N]
    """
    l, r, t, b = lrtb[:, 0], lrtb[:, 1], lrtb[:, 2], lrtb[:, 3]
    centerness = torch.sqrt(
        (torch.min(l, r) / (torch.max(l, r) + 1e-9)) *
        (torch.min(t, b) / (torch.max(t, b) + 1e-9))
    )
    return centerness  # 越靠近中心越接近 1

# 推理时最终分数 = cls_score * centerness，抑制边缘假阳性
final_score = cls_score * centerness_score
```

### FCOS 正样本分配规则

1. 落在 GT 框内的点视为候选正样本
2. 根据特征层尺度分配（小目标分配给高分辨率层）：P3 负责 ≤64px，P4 负责 64-128px，P5 负责 128-256px
3. 若点落在多个 GT 框内，选面积最小的 GT

```python
# 多层级分配示例（伪代码）
scale_ranges = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1e9)]
for level, (min_s, max_s) in enumerate(scale_ranges):
    # 筛选该层级负责的 GT 框
    valid_gt = (gt_sizes >= min_s) & (gt_sizes < max_s)
```

---

## CenterNet：Heatmap 生成与解码

CenterNet 将目标建模为中心点 heatmap，用高斯核渲染 GT 点。

### Heatmap 生成（高斯核渲染）

```python
import numpy as np

def gaussian_radius(det_size, min_overlap=0.7):
    """根据目标框大小计算高斯半径"""
    h, w = det_size
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)
    return int(min(r1))

def draw_gaussian(heatmap, center, radius, k=1):
    """在 heatmap 上以 center 为中心渲染高斯核"""
    diameter = 2 * radius + 1
    x = np.arange(0, diameter, 1, np.float32)
    y = x[:, np.newaxis]
    x0, y0 = radius, radius
    gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (radius / 3) ** 2))

    cx, cy = int(center[0]), int(center[1])
    H, W = heatmap.shape
    # 处理边界裁剪
    x1, x2 = max(0, cx - radius), min(W, cx + radius + 1)
    y1, y2 = max(0, cy - radius), min(H, cy + radius + 1)
    gx1, gx2 = radius - (cx - x1), radius + (x2 - cx)
    gy1, gy2 = radius - (cy - y1), radius + (y2 - cy)

    heatmap[y1:y2, x1:x2] = np.maximum(
        heatmap[y1:y2, x1:x2], gaussian[gy1:gy2, gx1:gx2]
    )
    return heatmap
```

### Heatmap 解码（推理阶段）

```python
def decode_heatmap(heatmap, wh, offset, K=100, score_threshold=0.3):
    """
    heatmap: [C, H, W] 各类别中心点热图
    wh:      [2, H, W] 宽高预测
    offset:  [2, H, W] 中心点偏移（弥补量化误差）
    K:       保留 Top-K 个检测结果
    """
    # 3×3 MaxPool 做 NMS（峰值点检测）
    hmax = F.max_pool2d(heatmap.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep

    # 取 Top-K 峰值
    scores, inds = heatmap.reshape(-1).topk(K)
    mask = scores > score_threshold
    inds = inds[mask]
    scores = scores[mask]

    cy = inds // heatmap.shape[-1]
    cx = inds % heatmap.shape[-1]
    # 加偏移修正量化误差
    cx_f = cx.float() + offset[0].reshape(-1)[inds]
    cy_f = cy.float() + offset[1].reshape(-1)[inds]
    w = wh[0].reshape(-1)[inds]
    h = wh[1].reshape(-1)[inds]
    # 还原到原图坐标（乘以下采样步长，如 4）
    boxes = torch.stack([cx_f - w/2, cy_f - h/2, cx_f + w/2, cy_f + h/2], dim=1) * 4
    return boxes, scores
```

---

## DETR：端到端目标检测完整流程

DETR 是第一个完全去除 NMS 的端到端检测器。

### 架构流程

```
输入图像 [B, 3, H, W]
   ↓ CNN 骨干（ResNet）提取特征 [B, C, H/32, W/32]
   ↓ 1×1 卷积降维到 d_model=256
   ↓ 展平 + 加 2D 位置编码 → [B, HW, 256]
   ↓ Transformer Encoder（6层）
   ↓ Transformer Decoder（6层，输入 N=100 个可学习 object query）
   ↓ FFN 分类头：[B, N, num_classes+1]
   ↓ FFN 回归头：[B, N, 4]（cx, cy, w, h，归一化到 [0,1]）
   ↓ 二分匹配（Hungarian Algorithm）
   ↓ 计算损失（匹配后的 CE + L1 + GIoU）
```

### 二分匹配损失

```python
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

def hungarian_match(pred_logits, pred_boxes, gt_labels, gt_boxes):
    """
    pred_logits: [N, num_classes+1]
    pred_boxes:  [N, 4]（归一化 cx,cy,w,h）
    gt_labels:   [M]
    gt_boxes:    [M, 4]
    """
    N, M = len(pred_logits), len(gt_labels)
    # 分类代价：负对数概率
    cls_cost = -pred_logits.softmax(-1)[:, gt_labels]  # [N, M]
    # L1 代价
    l1_cost = torch.cdist(pred_boxes, gt_boxes, p=1)   # [N, M]
    # GIoU 代价
    giou_cost = -generalized_box_iou(
        box_cxcywh_to_xyxy(pred_boxes),
        box_cxcywh_to_xyxy(gt_boxes)
    )  # [N, M]
    # 组合代价矩阵
    cost = 1.0 * cls_cost + 5.0 * l1_cost + 2.0 * giou_cost
    pred_idx, gt_idx = linear_sum_assignment(cost.detach().cpu().numpy())
    return pred_idx, gt_idx
```

**踩坑记录**：DETR 原版训练需要 500 epochs 才能收敛，Deformable DETR 将其降至 50 epochs。实际工程中更推荐 RT-DETR 或 DINO 检测器。

---

## 正样本分配策略对比

正样本分配是影响检测器性能的关键因素：

| 策略 | 代表 | 核心思想 | 优缺点 |
|------|------|---------|--------|
| **IoU 阈值** | Faster RCNN | IoU > 0.7 为正样本 | 简单，对小目标不友好 |
| **ATSS** | ATSS | 自适应选 Top-k 统计 | 无需调 anchor 参数 |
| **SimOTA** | YOLOx | 最优运输简化版，cost 最小 | 精度高，速度慢 |
| **TAL** | YOLOv8 | Task-Aligned，对齐分类和定位 | 稳定，工程常用 |

### ATSS（自适应训练样本选择）核心逻辑

```python
# 对每个 GT，在各 FPN 层各选 Top-k 个距离最近的候选 anchor（k=9）
# 计算这些候选 anchor 与 GT 的 IoU
# 正样本阈值 = mean(IoUs) + std(IoUs)（自适应）
# 落在 GT 框内且 IoU ≥ 阈值 → 正样本
threshold = iou_candidates.mean() + iou_candidates.std()
positive_mask = (iou_candidates >= threshold) & inside_gt_mask
```

---

## Deformable DETR：可变形注意力

标准 DETR 的自注意力计算复杂度为 O(H²W²)，对高分辨率特征图代价极高。Deformable DETR 提出**可变形注意力**：

- 每个 query 仅关注**少量采样点**（默认 4 个参考点 × 8 个头 = 32 个点）
- 采样位置由网络**动态预测偏移量**，而非固定网格

```
标准注意力：query 与所有 K 个 key 做点积 → O(K) 复杂度
可变形注意力：query 预测 M 个偏移量 → 双线性插值采样 → 加权求和 → O(M) 复杂度（M 远小于 K）
```

```python
# 可变形注意力伪代码
def deformable_attention(query, reference_points, value, sampling_offsets, attn_weights):
    """
    reference_points: [B, Q, L, 2]，每个 query 在各特征层的参考点归一化坐标
    sampling_offsets: [B, Q, n_heads, L*n_points, 2]，预测偏移
    attn_weights:     [B, Q, n_heads, L*n_points]
    """
    # 采样坐标 = 参考点 + 偏移
    sampling_locs = reference_points + sampling_offsets
    # 双线性插值从 value 中采样
    sampled = bilinear_sample(value, sampling_locs)  # [B, Q, n_heads, L*n_points, d]
    # 加权求和
    output = (sampled * attn_weights.unsqueeze(-1)).sum(-2)
    return output
```

---

## Anchor-Free vs Anchor-Based

| 维度 | Anchor-Based | Anchor-Free |
|------|-------------|------------|
| 设计复杂度 | 需手动设计 anchor | 更简洁 |
| 超参敏感 | anchor 尺寸敏感 | 较低 |
| 小目标 | FPN 配合较好 | 需专门设计 |
| 趋势 | 逐渐被取代 | 主流方向 |
