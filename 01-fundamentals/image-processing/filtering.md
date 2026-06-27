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
blur     = cv2.GaussianBlur(img, (5, 5), 0)
median   = cv2.medianBlur(img, 5)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
```

## 频域滤波

- 傅里叶变换：`cv2.dft` / `np.fft.fft2`
- 低通滤波 → 平滑；高通滤波 → 锐化/边缘增强

---

## 卷积原理

### 什么是卷积

卷积（Convolution）是滤波的核心操作：将一个小的滤波器核（kernel）在图像上滑动，对每个位置的邻域像素进行加权求和。

```
(I * K)[i,j] = Σ_m Σ_n I[i-m, j-n] × K[m,n]
```

### ASCII 示意图

```
输入图像（局部）        滤波核 K (3×3)      输出
┌───┬───┬───┐        ┌───┬───┬───┐
│ 1 │ 2 │ 3 │        │ 1 │ 0 │-1 │
├───┼───┼───┤    ×   ├───┼───┼───┤  →  对应输出像素
│ 4 │ 5 │ 6 │        │ 2 │ 0 │-2 │     = Σ(逐元素乘积)
├───┼───┼───┤        ├───┼───┼───┤
│ 7 │ 8 │ 9 │        │ 1 │ 0 │-1 │
└───┴───┴───┘        └───┴───┴───┘

输出 = 1×1 + 2×0 + 3×(-1)
     + 4×2 + 5×0 + 6×(-2)
     + 7×1 + 8×0 + 9×(-1)
     = 1 - 3 + 8 - 12 + 7 - 9 = -8
（Sobel X 方向，检测垂直边缘）
```

---

## 各滤波器核的数值示例

### 均值滤波核（3×3）

```
1/9 × [1 1 1]
      [1 1 1]
      [1 1 1]
```

所有权重相等，等价于简单平均。

### 高斯滤波核（3×3，σ=1）

```
1/16 × [1 2 1]
       [2 4 2]
       [1 2 1]
```

中心权重最大，呈高斯分布，平滑效果更自然。

### Sobel 核

X 方向（检测垂直边缘）：
```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

Y 方向（检测水平边缘）：
```
[-1 -2 -1]
[ 0  0  0]
[ 1  2  1]
```

### Laplacian 核

```
[ 0  1  0]
[ 1 -4  1]
[ 0  1  0]
```

对边缘两侧响应为负，中心为正，可用于锐化：`sharpened = img + α * laplacian`

### Scharr 核（比 Sobel 更精确）

X 方向：
```
[ -3  0   3]
[-10  0  10]
[ -3  0   3]
```

```python
import numpy as np
import cv2

# 手动定义滤波核并应用
kernel_sharpen = np.array([[ 0, -1,  0],
                            [-1,  5, -1],
                            [ 0, -1,  0]], dtype=np.float32)
sharpened = cv2.filter2D(img, -1, kernel_sharpen)

# 使用 Scharr 算子
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
magnitude = np.sqrt(scharrx**2 + scharry**2)
```

---

## 边缘填充方式对比

卷积时边缘像素缺少邻居，有以下几种处理方式：

| 填充方式 | 说明 | OpenCV 对应 |
|---------|------|------------|
| 零填充（Zero/Constant） | 边缘外补 0 | `cv2.BORDER_CONSTANT` |
| 反射填充（Reflect） | 边缘镜像对称 | `cv2.BORDER_REFLECT` |
| 边缘复制（Replicate） | 复制最边缘像素 | `cv2.BORDER_REPLICATE` |
| 循环填充（Wrap） | 图像周期延拓 | `cv2.BORDER_WRAP` |

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg", 0)

# 手动填充（pad=2）后再滤波
padded_reflect   = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REFLECT)
padded_replicate = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
padded_zero      = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

# 效果对比：零填充会在边缘产生黑色条纹，反射填充更自然
```

---

## 频域滤波（NumPy FFT）

频域滤波将图像转到频率空间，在那里乘以滤波器掩码，再反变换回来。低频对应缓变区域（背景），高频对应边缘和纹理。

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg", 0).astype(float)

# 1. 傅里叶变换
F = np.fft.fft2(img)
Fshift = np.fft.fftshift(F)  # 将低频移到中心

# 2. 可视化频谱
magnitude_spectrum = 20 * np.log(np.abs(Fshift) + 1)

# 3. 低通滤波（圆形掩码）
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
radius = 30  # 只保留低频（半径内）

mask_lp = np.zeros((rows, cols), dtype=np.uint8)
cv2.circle(mask_lp, (ccol, crow), radius, 1, -1)

# 4. 高通滤波（取反）
mask_hp = 1 - mask_lp

# 5. 应用并反变换
def apply_filter(Fshift, mask):
    F_filtered = Fshift * mask
    F_ishift   = np.fft.ifftshift(F_filtered)
    img_back   = np.abs(np.fft.ifft2(F_ishift))
    return img_back

img_lp = apply_filter(Fshift, mask_lp)  # 低通：平滑效果
img_hp = apply_filter(Fshift, mask_hp)  # 高通：边缘效果

# 6. 可视化
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, data, title in zip(axes,
        [img, magnitude_spectrum, img_lp, img_hp],
        ['原图', '频谱', '低通（平滑）', '高通（边缘）']):
    ax.imshow(data, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### 带通滤波器（同态滤波增强对比度）

```python
def homomorphic_filter(img_bgr, low=0.3, high=1.5, cutoff=30):
    """
    同态滤波：在对数域进行高通滤波，压缩光照变化，增强纹理对比度
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    # 取对数（分离照明和反射）
    img_log = np.log1p(img_gray)
    # 傅里叶变换
    F = np.fft.fftshift(np.fft.fft2(img_log))
    # 高斯高通滤波器
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
    D = np.sqrt(u**2 + v**2)
    H = (high - low) * (1 - np.exp(-D**2 / (2 * cutoff**2))) + low
    # 应用并反变换
    F_filtered = F * H
    img_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
    # 指数还原
    result = np.expm1(img_filtered)
    return np.clip(result, 0, 255).astype(np.uint8)
```

---

## 参考资料

- 《数字图像处理》Gonzalez & Woods（第4版）
- OpenCV 文档：Image Filtering
- NumPy FFT 文档：https://numpy.org/doc/stable/reference/routines.fft.html
