# 图像滤波

## 常用滤波器

| 滤波器 | 作用 | 核心参数 |
|--------|------|---------|
| 均值滤波 | 平滑，去噪 | 核大小 |
| 高斯滤波 | 平滑，保边优于均值 | 核大小、sigma |
| 中值滤波 | 去椒盐噪声 | 核大小 |
| 双边滤波 | 保边平滑 | d、sigmaColor、sigmaSpace |
| Sobel | 求梯度（边缘） | ksize、方向 |
| Laplacian | 二阶导数，锐化/边缘 | ksize |

## OpenCV 示例

```python
blur = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
```

## 频域滤波

- 傅里叶变换：`cv2.dft` / `np.fft.fft2`
- 低通滤波 → 平滑；高通滤波 → 锐化/边缘增强
