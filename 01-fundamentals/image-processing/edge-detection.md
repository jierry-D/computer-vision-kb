# 边缘检测

## 方法对比

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| Sobel | 一阶差分 | 快速，抗噪 | 边缘粗 |
| Scharr | 改进 Sobel | 精度更高 | — |
| Laplacian | 二阶差分 | 各向同性 | 对噪声敏感 |
| Canny | 多步骤最优 | 细边缘，双阈值 | 参数较多 |

## Canny 流程

1. 高斯平滑（去噪）
2. 计算梯度幅值与方向
3. 非极大值抑制（NMS）
4. 双阈值筛选（高阈值确定强边缘，低阈值延伸弱边缘）
5. 边缘连接

## OpenCV 示例

```python
edges  = cv2.Canny(img, threshold1=50, threshold2=150)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
```

---

## Canny 算法 NumPy 逐步实现

以下是 Canny 边缘检测的完整 NumPy 实现，帮助理解每一步原理。

```python
import numpy as np
import cv2

def canny_from_scratch(img, low_threshold=50, high_threshold=150, sigma=1.4):
    """
    Canny 边缘检测完整实现
    """
    # ─── Step 1: 高斯平滑 ───
    ksize = int(6 * sigma + 1) | 1  # 奇数核
    smoothed = cv2.GaussianBlur(img.astype(float), (ksize, ksize), sigma)

    # ─── Step 2: Sobel 梯度 ───
    Gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx) * 180 / np.pi  # 单位：度
    direction[direction < 0] += 180  # 映射到 0-180 度

    # ─── Step 3: 非极大值抑制（NMS） ───
    H, W = magnitude.shape
    nms = np.zeros_like(magnitude)
    for i in range(1, H-1):
        for j in range(1, W-1):
            angle = direction[i, j]
            m = magnitude[i, j]
            # 根据梯度方向确定比较的两个邻居
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                p, q = magnitude[i, j+1], magnitude[i, j-1]   # 水平
            elif 22.5 <= angle < 67.5:
                p, q = magnitude[i+1, j-1], magnitude[i-1, j+1]  # 斜45度
            elif 67.5 <= angle < 112.5:
                p, q = magnitude[i+1, j], magnitude[i-1, j]    # 垂直
            else:  # 112.5 <= angle < 157.5
                p, q = magnitude[i-1, j-1], magnitude[i+1, j+1]  # 斜135度
            nms[i, j] = m if (m >= p and m >= q) else 0

    # ─── Step 4: 双阈值筛选 ───
    strong_edges = (nms >= high_threshold)
    weak_edges   = (nms >= low_threshold) & (nms < high_threshold)

    # ─── Step 5: 边缘连接（滞后阈值）───
    result = np.zeros_like(magnitude, dtype=np.uint8)
    result[strong_edges] = 255

    # 若弱边缘的 8 邻域内有强边缘，则保留
    from scipy.ndimage import label
    weak_img = weak_edges.astype(np.uint8) * 255
    for i in range(1, H-1):
        for j in range(1, W-1):
            if weak_edges[i, j]:
                if strong_edges[i-1:i+2, j-1:j+2].any():
                    result[i, j] = 255

    return result

# 与 OpenCV 对比验证
img = cv2.imread("image.jpg", 0)
canny_scratch = canny_from_scratch(img, 50, 150)
canny_cv      = cv2.Canny(img, 50, 150)
print("差异像素数:", np.sum(canny_scratch != canny_cv))
```

---

## LoG（Laplacian of Gaussian）

LoG 先用高斯平滑去噪，再用 Laplacian 检测边缘，是对直接使用 Laplacian 对噪声敏感问题的改进。

### 原理

```
LoG(x,y) = ∇²G(x,y) = (x²+y²-2σ²)/(σ⁴) × exp(-(x²+y²)/(2σ²))
```

因其形状像墨西哥帽，也称 **Mexican Hat 算子**。

**边缘位置** = LoG 响应的**零交叉点（Zero Crossing）**

```python
import numpy as np
import cv2

def log_edge_detection(img, sigma=1.5, threshold=10):
    """
    LoG 边缘检测：高斯平滑 + Laplacian + 零交叉点检测
    """
    # Step 1: 高斯平滑
    ksize = int(6 * sigma + 1) | 1
    smoothed = cv2.GaussianBlur(img.astype(float), (ksize, ksize), sigma)

    # Step 2: Laplacian
    laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)

    # Step 3: 零交叉点检测
    H, W = laplacian.shape
    zero_crossings = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, H-1):
        for j in range(1, W-1):
            patch = laplacian[i-1:i+2, j-1:j+2]
            # 若邻域内同时有正值和负值，则为零交叉点
            if patch.min() < -threshold and patch.max() > threshold:
                zero_crossings[i, j] = 255

    return zero_crossings

# 直接使用 OpenCV（近似）
img = cv2.imread("image.jpg", 0)
blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
log_result = cv2.Laplacian(blurred, cv2.CV_64F)
log_abs = np.uint8(np.absolute(log_result))
```

---

## Harris 角点检测

### 原理

角点是两个方向梯度都很大的点。Harris 使用 **结构张量（Structure Tensor）** M 分析局部梯度分布：

```
M = Σ w(x,y) × [Ix²    IxIy]
                [IxIy   Iy² ]
```

角点响应函数：

```
R = det(M) - k × trace(M)²
  = λ₁λ₂ - k(λ₁+λ₂)²
```

- 若 R >> 0：**角点**（两个特征值都大）
- 若 R << 0：**边缘**（一大一小）
- 若 |R| ≈ 0：**平坦区域**（两个特征值都小）

```python
import numpy as np
import cv2

def harris_corner_detection(img, k=0.04, block_size=3, ksize=3, threshold=0.01):
    """
    Harris 角点检测完整实现
    """
    img_float = img.astype(np.float32)

    # Step 1: 计算梯度
    Ix = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=ksize)

    # Step 2: 计算结构张量分量
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Step 3: 高斯加权（局部窗口）
    ksize_blur = (block_size * 2 + 1, block_size * 2 + 1)
    Sxx = cv2.GaussianBlur(Ixx, ksize_blur, 0)
    Syy = cv2.GaussianBlur(Iyy, ksize_blur, 0)
    Sxy = cv2.GaussianBlur(Ixy, ksize_blur, 0)

    # Step 4: 计算角点响应 R
    det_M   = Sxx * Syy - Sxy**2
    trace_M = Sxx + Syy
    R = det_M - k * trace_M**2

    # Step 5: 阈值筛选 + 非极大值抑制
    R_norm = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
    corners_mask = (R_norm > threshold * 255).astype(np.uint8)

    # 使用 dilate 做简单 NMS（局部极大值）
    dilated = cv2.dilate(R_norm, None)
    corners = (R_norm == dilated) & (corners_mask == 1)

    return corners, R

# OpenCV 内置 Harris
img_gray = cv2.imread("image.jpg", 0)
dst = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
# 标记角点（红色）
img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
img_color[dst > 0.01 * dst.max()] = [0, 0, 255]

# 用亚像素精化
corners = np.argwhere(dst > 0.01 * dst.max()).astype(np.float32)
if len(corners) > 0:
    corners_refined = cv2.cornerSubPix(
        img_gray.astype(np.float32), corners[:, ::-1],
        winSize=(5, 5), zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
```

---

## 参考资料

- 《计算机视觉：算法与应用》Szeliski（边缘与角点检测章节）
- OpenCV 文档：Feature Detection
- John Canny 原始论文（1986）：A Computational Approach to Edge Detection
- Harris & Stephens（1988）：A Combined Corner and Edge Detector
