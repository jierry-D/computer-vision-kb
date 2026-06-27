# Anchor-Based 目标检测

## 核心思想

预设固定的 anchor box（不同尺寸+宽高比），网络预测相对 anchor 的偏移量和置信度。

## 主流模型

### 两阶段（Two-Stage）

| 模型 | 年份 | 特点 |
|------|------|------|
| R-CNN | 2014 | 开山之作，速度慢 |
| Fast R-CNN | 2015 | ROI Pooling，共享卷积特征 |
| Faster R-CNN | 2015 | RPN 生成候选框，端到端 |
| Mask R-CNN | 2017 | 加分割头，ROI Align |
| Cascade R-CNN | 2018 | 级联检测头，逐步精化 |

### 单阶段（One-Stage）

| 模型 | 年份 | 特点 |
|------|------|------|
| SSD | 2016 | 多尺度特征图 |
| YOLOv1-v3 | 2016-2018 | 速度极快，工业常用 |
| YOLOv4/v5 | 2020 | 工程优化，主流落地方案 |
| YOLOv8 | 2023 | Ultralytics，易用性极强 |
| PP-YOLOE | 2022 | 百度，工业场景强 |

---

## Faster R-CNN 完整流程

### 整体架构

```
输入图像
   ↓
骨干网络（ResNet + FPN）提取特征图
   ↓
RPN（Region Proposal Network）生成候选框
   ↓
ROI Pooling（从特征图裁剪固定大小特征）
   ↓
全连接分类头（类别分类 + 边界框回归）
   ↓
输出检测结果
```

### 第一阶段：RPN

RPN 在特征图每个位置生成多个 anchor（默认 3 种尺度 × 3 种宽高比 = 9 个），预测每个 anchor 的：
- **前景/背景二分类分数**（是否包含目标）
- **4 个偏移量**（相对 anchor 的 dx, dy, dw, dh）

正样本定义：与任意 GT 框 IoU ≥ 0.7，或 IoU 最大的 anchor。  
负样本定义：与所有 GT 框 IoU < 0.3。

```python
# RPN 损失（分类 + 回归）
rpn_cls_loss = F.binary_cross_entropy_with_logits(pred_cls, target_cls)
rpn_reg_loss = smooth_l1_loss(pred_delta, target_delta, reduction='sum') / N_pos
```

### 第二阶段：ROI Pooling

将不同大小的 RoI 统一映射到固定大小（如 7×7）的特征图：

```python
# torchvision 中的 ROI Pooling
from torchvision.ops import roi_pool, roi_align

# rois shape: [K, 5]，每行为 [batch_idx, x1, y1, x2, y2]
pooled = roi_pool(feature_map, rois, output_size=(7, 7), spatial_scale=1/16)
# 或使用精度更高的 ROI Align（双线性插值，无量化误差）
pooled = roi_align(feature_map, rois, output_size=(7, 7), spatial_scale=1/16)
```

### 第三阶段：分类与回归

```python
class RCNNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes + 1)   # +1 背景
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.cls_score(x), self.bbox_pred(x)
```

---

## FPN（Feature Pyramid Network）特征融合

FPN 通过**自顶向下路径 + 侧向连接**融合多尺度特征：

```
C5（最深，语义强）→ 上采样 → + → P5
C4             → 1×1 卷积 → + → P4
C3             → 1×1 卷积 → + → P3
C2             → 1×1 卷积 → + → P2
```

```python
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        # 侧向连接（lateral connections）：1×1 卷积统一通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        # 输出卷积：3×3 消除上采样混叠
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        # features: [C2, C3, C4, C5]，从浅到深
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest'
            )
        # 输出各尺度特征
        return [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
```

**注意事项**：上采样推荐用 `nearest` 而非 `bilinear`，速度更快且实验效果相近。

---

## YOLOv8 架构简介

YOLOv8 由 Ultralytics 于 2023 年发布，抛弃 anchor 思路（改为 anchor-free），但整体是 anchor-based 工程传统的延续。

### 主要模块

| 模块 | 说明 |
|------|------|
| **CSPNet 骨干** | 跨阶段部分网络，减少计算量同时保持精度 |
| **C2f 模块** | Cross Stage Partial with 2 bottlenecks，融合梯度流 |
| **SPPF** | 快速空间金字塔池化，替代 SPP |
| **解耦检测头** | 分类与定位分支独立，提升收敛速度 |

```python
# YOLOv8 快速使用
from ultralytics import YOLO

model = YOLO('yolov8n.pt')          # 加载预训练模型
results = model.predict('image.jpg', conf=0.5)  # 推理
model.train(data='coco.yaml', epochs=100)        # 训练
model.export(format='onnx')                      # 导出 ONNX
```

---

## NMS 与 Soft-NMS

### 标准 NMS

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: [N, 4]，格式 (x1, y1, x2, y2)
    scores: [N]
    返回保留的索引列表
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # 按分数降序
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算与剩余框的 IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep
```

### Soft-NMS

标准 NMS 会直接删除 IoU 超阈值的框，Soft-NMS 改为按 IoU 衰减分数，对密集场景更友好：

```python
def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.3):
    """高斯衰减版 Soft-NMS"""
    N = len(scores)
    scores = scores.copy()
    keep = []

    for i in range(N):
        max_idx = scores[i:].argmax() + i
        boxes[i], boxes[max_idx] = boxes[max_idx].copy(), boxes[i].copy()
        scores[i], scores[max_idx] = scores[max_idx], scores[i]

        if scores[i] < score_threshold:
            break
        keep.append(i)

        for j in range(i + 1, N):
            iou = compute_iou(boxes[i], boxes[j])
            # 高斯衰减
            scores[j] *= np.exp(-(iou ** 2) / sigma)
    return keep
```

**踩坑记录**：Soft-NMS 对有遮挡的密集行人效果明显优于 NMS，但会略微增加 FP；工业部署时可先用标准 NMS 快速验证，再用 Soft-NMS 精调。

---

## mAP 计算详解

mAP（mean Average Precision）是目标检测最重要的评估指标。

### 计算步骤

1. 对所有预测框按置信度降序排列
2. 依次标记为 TP / FP（IoU ≥ 阈值且未被占用的最大 IoU GT 框 → TP）
3. 累计计算 Precision 和 Recall，绘制 PR 曲线
4. 计算 AP（PR 曲线下面积，使用 11 点插值或全点积分）
5. 对所有类别取均值得到 mAP

```python
import numpy as np

def compute_ap(recalls, precisions):
    """计算单类 AP（全点积分法，COCO 风格）"""
    # 在首尾各补一个点
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    # 使精度单调不递减
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # 找 recall 变化的点
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_map(all_detections, all_gt, num_classes, iou_threshold=0.5):
    """
    all_detections: {class_id: [(image_id, score, x1,y1,x2,y2), ...]}
    all_gt:         {class_id: {image_id: [(x1,y1,x2,y2), ...]}}
    """
    aps = []
    for cls in range(num_classes):
        dets = sorted(all_detections.get(cls, []), key=lambda x: -x[1])
        gts = all_gt.get(cls, {})
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        n_gt = sum(len(v) for v in gts.values())
        matched = {img_id: [False] * len(boxes)
                   for img_id, boxes in gts.items()}

        for d_idx, (img_id, score, *box) in enumerate(dets):
            gt_boxes = gts.get(img_id, [])
            best_iou, best_j = 0, -1
            for j, gt_box in enumerate(gt_boxes):
                iou = compute_iou(box, gt_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and not matched[img_id][best_j]:
                tp[d_idx] = 1
                matched[img_id][best_j] = True
            else:
                fp[d_idx] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / (n_gt + 1e-9)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-9)
        aps.append(compute_ap(recalls, precisions))

    return np.mean(aps)
```

---

## Anchor 设计建议：K-Means 聚类

手动设计 anchor 尺寸容易与数据分布不匹配，建议用 K-Means 在数据集上聚类：

```python
import numpy as np

def kmeans_anchors(boxes_wh, k=9, iou_threshold=0.25, n_iter=1000):
    """
    boxes_wh: [N, 2]，归一化到 [0, 1] 的 (width, height)
    返回 k 个 anchor (w, h)
    """
    # 随机初始化
    idx = np.random.choice(len(boxes_wh), k, replace=False)
    clusters = boxes_wh[idx].copy()

    for _ in range(n_iter):
        # 计算每个框与所有 anchor 的 IoU（仅考虑尺寸，不考虑位置）
        d = 1 - box_iou_wh(boxes_wh, clusters)  # [N, k]
        assignments = d.argmin(axis=1)

        new_clusters = np.array([
            boxes_wh[assignments == i].mean(axis=0) for i in range(k)
        ])
        if np.allclose(new_clusters, clusters):
            break
        clusters = new_clusters

    avg_iou = (1 - d.min(axis=1)).mean()
    print(f"平均 IoU: {avg_iou:.4f}")
    # 按面积排序
    areas = clusters[:, 0] * clusters[:, 1]
    return clusters[areas.argsort()]

def box_iou_wh(wh1, wh2):
    """仅基于宽高计算 IoU（假设框均以中心对齐）"""
    inter_w = np.minimum(wh1[:, None, 0], wh2[None, :, 0])
    inter_h = np.minimum(wh1[:, None, 1], wh2[None, :, 1])
    inter = inter_w * inter_h
    area1 = wh1[:, 0] * wh1[:, 1]
    area2 = wh2[:, 0] * wh2[:, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union
```

**注意事项**：
- 聚类前记得将 bbox 宽高归一化到 [0,1]（除以图像宽高）
- YOLOv5 默认使用 IoU 作为距离度量，而非欧氏距离
- 聚类结果仅作为初始化，训练中 anchor 不更新（可通过 offset 回归隐式适应）

---

## FPN（特征金字塔网络）

- 融合多层特征，解决多尺度目标检测问题
- 几乎所有现代检测器都采用 FPN 结构

## 评估指标

- **mAP**（mean Average Precision）：各类 AP 均值
- **AP@0.5**：IoU 阈值 0.5 时的精度
- **AP@0.5:0.95**：COCO 标准，多 IoU 阈值均值
- **FPS**：推理帧率
