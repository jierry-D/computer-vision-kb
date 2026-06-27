# 色彩空间

## 常用色彩空间

| 色彩空间 | 特点 | 典型用途 |
|---------|------|---------|
| RGB | 显示标准，三通道 | 网络输入、显示 |
| BGR | OpenCV 默认 | OpenCV 读图 |
| HSV | 色调/饱和度/亮度分离 | 颜色分割、阈值 |
| LAB | 感知均匀，亮度分离 | 颜色归一化 |
| YUV/YCbCr | 亮度与色度分离 | 视频压缩 |
| 灰度 | 单通道 | 边缘检测、形态学 |

## OpenCV 转换

```python
import cv2
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
```

## 注意事项

- OpenCV 读图默认 BGR，PyTorch/PIL 默认 RGB，注意转换
- HSV 中 H 范围在 OpenCV 里是 0-180（非 0-360）
