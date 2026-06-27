# 深度估计

## 方法分类

### 主动式
- **结构光**（RealSense D435、Kinect）：投影已知图案，测量变形量
- **ToF（飞行时间）**：测量光传播时间，分辨率较低

### 被动式

| 方法 | 输入 | 代表 |
|------|------|------|
| 双目立体匹配 | 左右图像 | SGBM、PSMNet、RAFT-Stereo |
| 单目深度估计 | 单张图像 | MiDaS、Depth Anything、Marigold |
| 多视图 SfM | 多张图像 | COLMAP |

## 单目深度估计（主流方向）

### MiDaS
- 跨数据集训练，相对深度估计
- 适合通用场景

### Depth Anything V2
- 目前最强单目深度，支持绝对/相对深度

```python
from transformers import pipeline
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
depth = pipe(image)["depth"]
```

## 评估指标

- AbsRel、SqRel、RMSE、δ<1.25
