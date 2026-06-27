# 实例分割

> 同时检测目标并生成逐实例的像素级掩码，区分同类不同实例。

## 主流方法

### Two-Stage（先检测后分割）

| 模型 | 特点 |
|------|------|
| Mask R-CNN | 实例分割开山之作，ROI Align + 掩码头 |
| PANet | Path Aggregation，特征融合改进 |
| HTC | 级联检测+分割，精度高 |

### One-Stage

| 模型 | 特点 |
|------|------|
| YOLACT | 实时实例分割，线性组合掩码 |
| SOLOv2 | 按位置分类，无需检测框 |
| CondInst | 动态卷积核生成掩码 |

### Query-Based

| 模型 | 特点 |
|------|------|
| QueryInst | 基于 DETR 的实例分割 |
| Mask2Former | 统一框架，query 匹配 |
| SAM | Meta 通用分割大模型，zero-shot |

## SAM（Segment Anything Model）

```python
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)
masks, scores, logits = predictor.predict(point_coords=..., point_labels=...)
```

## 评估指标

- **mask AP**：基于掩码 IoU 的 AP（COCO 标准）
