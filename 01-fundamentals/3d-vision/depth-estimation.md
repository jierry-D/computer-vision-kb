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

---

## 主流深度传感器规格对比

| 传感器 | 技术原理 | 深度范围 | 精度 | 分辨率 | 帧率 | 价格区间 | 备注 |
|--------|----------|----------|------|--------|------|----------|------|
| Intel RealSense D435 | 结构光 + 双目 | 0.3~3m | ~2mm @1m | 1280×720 | 90fps | $200~400 | 室内首选，户外受光照影响 |
| Intel RealSense D455 | 结构光 + 双目 | 0.6~6m | ~2mm @1m | 1280×720 | 90fps | $300~500 | 更大基线，远距更准 |
| Microsoft Azure Kinect | ToF | 0.5~3.86m | ~1.1mm @1m | 640×576 | 30fps | $400 | 高精度，含 RGB+IMU |
| Orbbec Gemini 2 | 结构光 | 0.15~8m | <1% @1m | 1280×800 | 30fps | $200 | 国产，ROS 支持好 |
| FLIR Blackfly（双目）| 被动双目 | 取决于基线 | 相对精度高 | 多种 | 100fps+ | $500+ | 工业检测 |
| Velodyne VLP-16 | LiDAR ToF | 1~100m | ±3cm | 16线 | 20fps | $4000+ | 自动驾驶 |
| Livox Mid-360 | LiDAR | 0.1~40m | ±2cm | 非重复扫描 | — | $1000+ | 小型化 LiDAR |

---

## 单目深度估计评估代码

标准指标（以 KITTI 或 NYU Depth 为基准）：

```python
import numpy as np
import torch

def compute_depth_errors(pred, gt, min_depth=0.1, max_depth=80.0):
    """
    pred, gt: numpy array，单位为米
    返回常用评估指标字典
    """
    mask = (gt > min_depth) & (gt < max_depth)
    pred = pred[mask]
    gt   = gt[mask]

    thresh = np.maximum(pred / gt, gt / pred)
    delta1 = (thresh < 1.25).mean()       # δ < 1.25
    delta2 = (thresh < 1.25**2).mean()    # δ < 1.25²
    delta3 = (thresh < 1.25**3).mean()    # δ < 1.25³

    abs_rel = np.abs(pred - gt) / gt              # AbsRel（越小越好）
    sq_rel  = ((pred - gt) ** 2) / gt             # SqRel
    rmse    = np.sqrt(((pred - gt) ** 2).mean())  # RMSE
    rmse_log = np.sqrt((np.log(pred) - np.log(gt)) ** 2).mean()  # RMSE log

    return {
        'AbsRel':   abs_rel.mean(),
        'SqRel':    sq_rel.mean(),
        'RMSE':     rmse,
        'RMSElog':  rmse_log,
        'delta1':   delta1,
        'delta2':   delta2,
        'delta3':   delta3,
    }

# 使用示例
pred_depth = np.load('pred.npy')   # (H, W)，预测深度
gt_depth   = np.load('gt.npy')     # (H, W)，真实深度

# 尺度对齐（单目相对深度需要先对齐尺度）
scale = np.median(gt_depth[gt_depth > 0]) / np.median(pred_depth[pred_depth > 0])
pred_depth *= scale

metrics = compute_depth_errors(pred_depth, gt_depth)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
```

---

## RealSense 相机 Python SDK 使用示例

```python
import pyrealsense2 as rs
import numpy as np
import cv2

def get_realsense_frames():
    """RealSense D435 采集 RGB + 深度图"""
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置流：RGB 30fps + 深度 30fps
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 开始采集
    profile = pipeline.start(config)

    # 获取深度单位（通常为 0.001，即 1mm）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度单位: {depth_scale} 米/单位")

    # 深度-RGB 对齐（将深度图对齐到 RGB 视角）
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())  # uint16
            color_image = np.asanyarray(color_frame.get_data())  # uint8 BGR

            # 深度图转米
            depth_meters = depth_image * depth_scale

            # 获取相机内参
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.ppx, intrinsics.ppy

            yield color_image, depth_meters, (fx, fy, cx, cy)
    finally:
        pipeline.stop()


# 后处理滤波（RealSense 内置）
def apply_realsense_filters(depth_frame):
    """应用 RealSense 内置深度滤波器"""
    # 1. 抽取（降噪）
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)
    # 2. 空间滤波（去除边缘噪声）
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
    # 3. 时域滤波（帧间平滑）
    temporal = rs.temporal_filter()
    # 4. 孔洞填充
    hole_filling = rs.hole_filling_filter()

    filtered = decimation.process(depth_frame)
    filtered = spatial.process(filtered)
    filtered = temporal.process(filtered)
    filtered = hole_filling.process(filtered)
    return filtered
```

---

## 深度图滤波（孔洞填充/去噪）代码

```python
import numpy as np
import cv2
from scipy.ndimage import median_filter

def fill_depth_holes(depth, max_hole_size=50):
    """
    孔洞填充：对深度图中无效区域（depth==0）进行填充
    使用形态学膨胀迭代填充
    """
    filled = depth.copy().astype(np.float32)
    invalid_mask = filled == 0

    # 从最近邻有效深度扩散填充
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(max_hole_size):
        # 膨胀：将有效深度值扩散到相邻无效区域
        dilated = cv2.dilate(filled, kernel)
        filled[invalid_mask] = dilated[invalid_mask]
        invalid_mask = filled == 0
        if not invalid_mask.any():
            break
    return filled


def denoise_depth(depth, method='median', kernel_size=5):
    """
    深度图去噪
    method: 'median' / 'bilateral' / 'gaussian'
    """
    valid_mask = depth > 0
    if method == 'median':
        smoothed = median_filter(depth.astype(np.float32), size=kernel_size)
    elif method == 'bilateral':
        # 双边滤波：保留边缘的同时去噪
        smoothed = cv2.bilateralFilter(
            depth.astype(np.float32), d=kernel_size,
            sigmaColor=50, sigmaSpace=50)
    elif method == 'gaussian':
        smoothed = cv2.GaussianBlur(
            depth.astype(np.float32), (kernel_size, kernel_size), 0)
    # 只对有效区域应用滤波结果
    result = depth.copy().astype(np.float32)
    result[valid_mask] = smoothed[valid_mask]
    return result
```

---

## 深度图转彩色可视化代码

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

def depth_to_colormap(depth, min_depth=None, max_depth=None,
                       colormap=cv2.COLORMAP_TURBO):
    """
    将深度图映射为伪彩色图像
    depth: (H, W) float，单位为米
    返回: (H, W, 3) uint8 BGR
    """
    valid_mask = depth > 0
    d = depth.copy().astype(np.float32)

    if min_depth is None:
        min_depth = d[valid_mask].min() if valid_mask.any() else 0
    if max_depth is None:
        max_depth = d[valid_mask].max() if valid_mask.any() else 1

    # 归一化到 0~255
    d = np.clip((d - min_depth) / (max_depth - min_depth + 1e-6), 0, 1)
    d_uint8 = (d * 255).astype(np.uint8)

    # 应用颜色映射
    colored = cv2.applyColorMap(d_uint8, colormap)
    # 无效区域设为黑色
    colored[~valid_mask] = 0
    return colored


# 使用示例
depth = np.load('depth.npy')  # (H, W) float32，单位米
vis = depth_to_colormap(depth, min_depth=0.5, max_depth=5.0)
cv2.imwrite('depth_color.png', vis)

# 并排显示 RGB + 深度
rgb = cv2.imread('rgb.png')
combined = np.hstack([rgb, vis])
cv2.imwrite('combined.png', combined)

# Matplotlib 版本（研究展示）
plt.figure(figsize=(8, 4))
plt.imshow(depth, cmap='turbo', vmin=0.5, vmax=5.0)
plt.colorbar(label='Depth (m)')
plt.title('Depth Map')
plt.savefig('depth_vis.png', dpi=150)
```
