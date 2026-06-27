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

## FPN（特征金字塔网络）

- 融合多层特征，解决多尺度目标检测问题
- 几乎所有现代检测器都采用 FPN 结构

## 评估指标

- **mAP**（mean Average Precision）：各类 AP 均值
- **AP@0.5**：IoU 阈值 0.5 时的精度
- **AP@0.5:0.95**：COCO 标准，多 IoU 阈值均值
- **FPS**：推理帧率
