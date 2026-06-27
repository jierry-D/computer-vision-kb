# 全景分割

> 语义分割 + 实例分割的统一：stuff（天空/地面等无定形区域）+ things（可数目标）

## 评估指标

- **PQ（Panoptic Quality）** = SQ × RQ
  - SQ（Segmentation Quality）：匹配实例的平均 IoU
  - RQ（Recognition Quality）：F1 得分

## 主流方法

| 模型 | 特点 |
|------|------|
| Panoptic FPN | 两分支融合 |
| Panoptic-DeepLab | 自底向上，速度快 |
| MaskFormer | 统一语义/实例/全景，基于 Transformer |
| Mask2Former | MaskFormer 升级版，SOTA |
| OneFormer | 单模型三任务，条件 query |

## 与语义/实例分割的关系

```
语义分割：每像素类别（不分实例）
实例分割：每实例掩码（忽略背景类）
全景分割：= 语义分割（stuff）+ 实例分割（things）
```

---

## Panoptic FPN：两分支融合逻辑

Panoptic FPN 在 Mask R-CNN 基础上增加语义分割分支，最终合并两路输出：

```
FPN 特征图（P2~P5）
   ├─→ 实例分割分支（Mask R-CNN Head）→ 每个实例的 mask + 类别
   └─→ 语义分割分支（逐层上采样融合）→ 全图 stuff 语义标签
         ↓
   后处理合并：
     1. 优先填入 stuff（语义分割结果）
     2. 用 things 实例掩码覆盖对应区域
     3. 处理重叠冲突（按置信度选最高分实例）
```

```python
def merge_panoptic(sem_seg, instance_masks, instance_classes, instance_scores,
                   stuff_classes, overlap_threshold=0.5):
    """
    sem_seg:          [H, W]，语义分割预测（含 stuff 和 things 类）
    instance_masks:   [N, H, W]，实例掩码（bool）
    instance_classes: [N]，实例类别 ID
    instance_scores:  [N]，实例置信度
    stuff_classes:    stuff 类别 ID 集合
    """
    H, W = sem_seg.shape
    panoptic = np.zeros((H, W), dtype=np.int32)   # 存储 segment_id
    segments_info = []
    current_id = 1

    # 1. 填入 stuff 区域
    for cls in np.unique(sem_seg):
        if cls in stuff_classes:
            mask = sem_seg == cls
            panoptic[mask] = current_id
            segments_info.append({'id': current_id, 'category_id': int(cls),
                                   'isthing': False})
            current_id += 1

    # 2. 按置信度从高到低填入 things 实例
    order = instance_scores.argsort()[::-1]
    for idx in order:
        mask = instance_masks[idx]
        # 计算与已填充区域的重叠比例
        already_filled = panoptic[mask] != 0
        overlap = already_filled.sum() / (mask.sum() + 1e-9)
        if overlap > overlap_threshold:
            continue  # 大部分被占用，跳过
        # 只更新未填充区域
        panoptic[mask & (panoptic == 0)] = current_id
        segments_info.append({'id': current_id,
                               'category_id': int(instance_classes[idx]),
                               'isthing': True})
        current_id += 1

    return panoptic, segments_info
```

---

## Mask2Former：Masked Attention 机制

Mask2Former 的核心改进是将 cross-attention 限制在**预测 mask 内部**，显著提升收敛速度和精度。

### 标准 Cross-Attention vs Masked Attention

```
标准 Cross-Attention：
  每个 query 关注全图所有位置（H×W 个 key/value）

Masked Attention：
  每个 query 只关注其预测 mask 为正的位置
  mask 由上一 Decoder 层的输出获得（初始为全 1）
```

```python
def masked_cross_attention(query, key, value, mask):
    """
    query: [B, Q, d]  Q 个 object query
    key:   [B, HW, d] 特征图展平
    value: [B, HW, d]
    mask:  [B, Q, HW] bool，True 表示该位置可以被关注
    """
    # 注意力分数
    attn = torch.bmm(query, key.transpose(1, 2)) / (query.shape[-1] ** 0.5)  # [B, Q, HW]
    # 将不在 mask 内的位置设为极小值（softmax 后趋近于 0）
    attn = attn.masked_fill(~mask, float('-inf'))
    attn = attn.softmax(dim=-1)
    # 处理全 -inf 的行（初始 mask 可能全为 False）
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.bmm(attn, value)  # [B, Q, d]
```

### 训练技巧

- Mask2Former 在**每个 Decoder 层**都计算辅助损失（不仅最后一层）
- 使用**匈牙利二分匹配**将 N 个 query 与 M 个 GT 进行一对一匹配
- 损失 = 分类 CE + Mask BCE + Mask Dice Loss

---

## PQ 指标详细计算方法

```python
import numpy as np
from collections import defaultdict

def compute_pq(pred_panoptic, pred_segments, gt_panoptic, gt_segments,
               num_categories, ignore_label=0):
    """
    pred_panoptic: [H, W]，每个像素的 segment_id
    pred_segments: list of {'id': int, 'category_id': int}
    gt_panoptic:   [H, W]
    gt_segments:   list of {'id': int, 'category_id': int}
    """
    # 构建类别 → segment 集合的映射
    def build_cat_map(segments):
        cat_map = defaultdict(set)
        for seg in segments:
            cat_map[seg['category_id']].add(seg['id'])
        return cat_map

    pred_cat = build_cat_map(pred_segments)
    gt_cat   = build_cat_map(gt_segments)

    sq_sum = defaultdict(float)   # 各类 IoU 之和
    tp     = defaultdict(int)     # True Positive
    fp     = defaultdict(int)     # False Positive
    fn     = defaultdict(int)     # False Negative

    all_cats = set(pred_cat) | set(gt_cat)

    for cat in all_cats:
        pred_ids = pred_cat.get(cat, set())
        gt_ids   = gt_cat.get(cat, set())

        for gt_id in gt_ids:
            gt_mask = gt_panoptic == gt_id
            best_iou, best_pred_id = 0, None
            for pred_id in pred_ids:
                pred_mask  = pred_panoptic == pred_id
                intersection = (gt_mask & pred_mask).sum()
                union        = (gt_mask | pred_mask).sum()
                iou = intersection / (union + 1e-9)
                if iou > best_iou:
                    best_iou, best_pred_id = iou, pred_id

            if best_iou > 0.5:
                tp[cat]     += 1
                sq_sum[cat] += best_iou
                pred_ids.discard(best_pred_id)
            else:
                fn[cat] += 1

        fp[cat] += len(pred_ids)  # 未被匹配的预测 = FP

    # 计算各类 PQ
    pq_per_cat = {}
    for cat in all_cats:
        denom = tp[cat] + 0.5 * fp[cat] + 0.5 * fn[cat]
        sq = sq_sum[cat] / (tp[cat] + 1e-9)
        rq = tp[cat] / (denom + 1e-9)
        pq_per_cat[cat] = {'PQ': sq * rq, 'SQ': sq, 'RQ': rq,
                            'TP': tp[cat], 'FP': fp[cat], 'FN': fn[cat]}

    mean_pq = np.mean([v['PQ'] for v in pq_per_cat.values()])
    return mean_pq, pq_per_cat
```

---

## 注意事项

- **stuff vs things 划分**：COCO 中有 53 个 stuff 类（天空、草地等）和 80 个 things 类，两者不重叠，预处理时需分别处理
- **segment_id 编码**：COCO 全景格式将 panoptic ID 编码为 `R + G*256 + B*256*256`，读取时需转换
- **Mask2Former 推理**：模型输出的 query 数量通常远多于实际目标数，需按置信度过滤并做 Panoptic Merge 后处理
- **PQ 低于预期**：常见原因是 stuff 类边界不精确，可用 CRF 后处理优化边界
