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
img_hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
```

## 注意事项

- OpenCV 读图默认 BGR，PyTorch/PIL 默认 RGB，注意转换
- HSV 中 H 范围在 OpenCV 里是 0-180（非 0-360）

---

## 各色彩空间深度解析

### RGB / BGR

- **RGB**：红(R)绿(G)蓝(B) 三通道，8 bit 图像每通道范围 0-255
- **BGR**：OpenCV 默认读取顺序，与 RGB 通道顺序相反

```python
import cv2
import numpy as np

img_bgr = cv2.imread("image.jpg")          # BGR，shape: (H, W, 3)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # → RGB（供 matplotlib/PyTorch 使用）

# 或者直接翻转通道
img_rgb2 = img_bgr[:, :, ::-1]
```

### HSV（色调/饱和度/亮度）

HSV 将颜色感知分为三个独立维度，**非常适合颜色分割任务**：

- **H（Hue，色调）**：颜色种类，OpenCV 中范围 0-180（对应 0°-360°）
  - 红色：0-10 和 170-180
  - 绿色：35-85
  - 蓝色：100-130
- **S（Saturation，饱和度）**：颜色纯度，0=灰色，255=纯色
- **V（Value，明度）**：亮度，0=黑色，255=最亮

**实战：HSV 颜色分割（提取红色区域）**

```python
import cv2
import numpy as np

def extract_red(img_bgr):
    """提取图像中的红色区域"""
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 红色在 HSV 中跨越 0 度（需要两个范围）
    lower_red1 = np.array([0,   100, 100])
    upper_red1 = np.array([10,  255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask  = cv2.bitwise_or(mask1, mask2)

    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return mask, result

# 提取绿色（更简单，单一范围）
def extract_green(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35,  50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    return mask
```

### LAB（CIE L*a*b*）

- **L**：亮度（0=黑，100=白），与人眼感知线性相关
- **a**：绿-红轴（负=绿，正=红）
- **b**：蓝-黄轴（负=蓝，正=黄）

**优点**：感知均匀，颜色距离与人眼感知一致，适合颜色归一化和颜色迁移。

```python
import cv2
import numpy as np

def color_transfer_lab(source, target):
    """将 source 图像的颜色风格迁移到 target 图像"""
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(float)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(float)

    # 逐通道均值/方差对齐
    for ch in range(3):
        src_mean, src_std = src_lab[:,:,ch].mean(), src_lab[:,:,ch].std()
        tgt_mean, tgt_std = tgt_lab[:,:,ch].mean(), tgt_lab[:,:,ch].std()
        tgt_lab[:,:,ch] = (tgt_lab[:,:,ch] - tgt_mean) / (tgt_std + 1e-6) * src_std + src_mean

    tgt_lab = np.clip(tgt_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(tgt_lab, cv2.COLOR_LAB2BGR)
```

### YUV / YCbCr

- **Y**：亮度（Luminance），人眼最敏感
- **Cb/U**：蓝色色度偏差
- **Cr/V**：红色色度偏差

视频编码（H.264/HEVC）利用人眼对色度分辨率低的特性，对 Y 通道用全分辨率，对 Cb/Cr 进行 2x 下采样（4:2:0 格式），实现约 50% 的数据压缩。

```python
img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
y, u, v = cv2.split(img_yuv)

# 直方图均衡化只作用于亮度通道（不影响颜色）
y_eq = cv2.equalizeHist(y)
img_eq = cv2.cvtColor(cv2.merge([y_eq, u, v]), cv2.COLOR_YUV2BGR)
```

---

## Bayer 阵列（相机传感器原始格式）

### 原理

数码相机传感器每个像素只感应一种颜色（R、G 或 B），排列成 Bayer 阵列：

```
R G R G R G
G B G B G B
R G R G R G
G B G B G B
```

绿色像素占 50%（人眼对绿色最敏感），红蓝各 25%。

### 去马赛克（Demosaicing）

将 Bayer 原始数据恢复为 RGB 图像：

```python
import cv2

# 读取 RAW Bayer 图像（16bit 灰度图存储）
raw = cv2.imread("raw_image.png", cv2.IMREAD_GRAYSCALE)

# Bayer RGGB → BGR
img_bgr = cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)
# 其他格式：COLOR_BayerGB2BGR, COLOR_BayerGR2BGR, COLOR_BayerBG2BGR
```

---

## 色彩归一化方法

### 1. Min-Max 归一化（归到 [0,1]）

```python
img_norm = img.astype(float) / 255.0
```

### 2. ImageNet 标准化（深度学习最常用）

```python
import numpy as np

mean = np.array([0.485, 0.456, 0.406])  # RGB 均值
std  = np.array([0.229, 0.224, 0.225])  # RGB 标准差

img_float = img_rgb.astype(float) / 255.0
img_norm  = (img_float - mean) / std
```

### 3. CLAHE（自适应直方图均衡化）

在 LAB 空间对亮度通道做局部均衡，保留颜色的同时增强对比度：

```python
import cv2

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(img_lab)
l_clahe = clahe.apply(l)
img_clahe = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)
```

---

## 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|---------|
| BGR/RGB 混淆 | OpenCV 返回 BGR，Matplotlib/PIL/PyTorch 期望 RGB | 用 `cv2.cvtColor` 或 `[:,:,::-1]` 转换 |
| HSV 的 H 范围 | OpenCV 是 0-180，其他工具多为 0-360 | 注意阈值设置时对应缩放 |
| LAB 通道范围 | OpenCV uint8 时 L:[0,255], a/b:[0,255]（中心=128）；float 时 L:[0,100], a/b:[-127,127] | 明确数据类型后再运算 |
| YUV 与 YCbCr | 严格上是不同标准（BT.601 vs BT.709），但 OpenCV 统称 YCrCb | 视频处理时注意标准版本 |
| 颜色分割受光照影响 | HSV 的 V 通道受环境光影响 | 同时约束 S 通道（低饱和度为灰色/白色，排除） |
