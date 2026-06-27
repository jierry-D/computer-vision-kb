# 目标跟踪

## 任务分类

| 类型 | 说明 |
|------|------|
| SOT（单目标跟踪） | 给定初始框，跟踪整个序列 |
| MOT（多目标跟踪） | 同时跟踪视频中所有目标 |
| MOTS | 多目标跟踪 + 分割 |

## SOT 主流方法

| 模型 | 特点 |
|------|------|
| SiamFC | 孪生网络相似度匹配 |
| SiamRPN++ | 加 RPN，精度提升 |
| OSTrack | Transformer 一体化跟踪 |
| MixFormer | 混合注意力，SOTA |

## MOT 主流方法

| 模型 | 特点 |
|------|------|
| SORT | Kalman + 匈牙利算法，速度快 |
| DeepSORT | SORT + ReID 特征，抗 ID 切换 |
| ByteTrack | 低置信度检测框也参与匹配，SOTA |
| StrongSORT | DeepSORT 改进版 |
| BoT-SORT | ByteTrack + 相机运动补偿 |

## 评估指标（MOT）

- **MOTA**：多目标跟踪精度
- **MOTP**：多目标跟踪准确度
- **IDF1**：ID 一致性 F1
- **IDs**：ID 切换次数（越低越好）
