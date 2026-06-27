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
