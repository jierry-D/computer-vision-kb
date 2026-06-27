# 形态学操作

> 基于结构元素（kernel）对二值/灰度图进行形状分析。

## 基本操作

| 操作 | 效果 | 用途 |
|------|------|------|
| 腐蚀 Erosion | 缩小前景，去小噪点 | 去除小白点 |
| 膨胀 Dilation | 扩大前景，填小孔 | 连接断裂区域 |
| 开运算 Opening | 腐蚀→膨胀，去小目标 | 消除小噪声 |
| 闭运算 Closing | 膨胀→腐蚀，填小孔洞 | 填充内部空洞 |
| 梯度 Gradient | 膨胀-腐蚀 | 提取边缘 |
| 顶帽 Top Hat | 原图-开运算 | 提取亮细节 |
| 黑帽 Black Hat | 闭运算-原图 | 提取暗细节 |

## OpenCV 示例

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded  = cv2.erode(img, kernel, iterations=1)
dilated = cv2.dilate(img, kernel, iterations=1)
opened  = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closed  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```

---

## 结构元素形状对比

结构元素（Structuring Element）决定形态学操作的"探针"形状，不同形状适用于不同场景。

```python
import cv2
import numpy as np

# 矩形：各方向均等影响
rect  = cv2.getStructuringElement(cv2.MORPH_RECT,    (5, 5))
# 椭圆/圆形：对圆形目标更自然
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# 十字形：只在水平垂直方向扩展
cross = cv2.getStructuringElement(cv2.MORPH_CROSS,   (5, 5))
```

各形状示意（5×5）：

```
矩形（RECT）          椭圆（ELLIPSE）        十字（CROSS）
1 1 1 1 1            0 0 1 0 0             0 0 1 0 0
1 1 1 1 1            0 1 1 1 0             0 0 1 0 0
1 1 1 1 1            1 1 1 1 1             1 1 1 1 1
1 1 1 1 1            0 1 1 1 0             0 0 1 0 0
1 1 1 1 1            0 0 1 0 0             0 0 1 0 0
```

**选择建议**：
- 矩形：通用，计算快
- 椭圆：处理圆形/球形目标（细胞、硬币）
- 十字：细线段检测、水平/垂直方向分析

---

## 形态学重建（Morphological Reconstruction）

### 概念

形态学重建使用一幅**标记图像（Marker）**在**掩码图像（Mask）**约束下进行膨胀，直到稳定为止。与普通膨胀不同，它不会超出掩码范围。

**用途**：
- 去除接触边界的目标（孔洞填充）
- 提取内部连通区域
- 去除边缘相连的小目标

```python
import numpy as np
import cv2
from skimage.morphology import reconstruction

# 场景：填充二值图像中的内部孔洞
def fill_holes(binary_img):
    """
    填充二值图像内部的所有孔洞
    原理：从边界出发做形态学重建，未被重建到的区域即为内部孔洞
    """
    # 标记：从图像边界初始化
    marker = binary_img.copy()
    marker[1:-1, 1:-1] = 0  # 只保留边界

    # 在掩码约束下做重建（膨胀直到稳定）
    # skimage 中 reconstruction 执行此操作
    filled = reconstruction(marker, binary_img, method='dilation')
    return (filled > 0).astype(np.uint8) * 255

# OpenCV 实现：flood fill 方法
def fill_holes_cv(binary_img):
    img_copy = binary_img.copy()
    h, w = img_copy.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_copy, mask, (0, 0), 255)
    img_floodfill_inv = cv2.bitwise_not(img_copy)
    filled = binary_img | img_floodfill_inv
    return filled
```

---

## 骨架提取（Skeletonization）

骨架提取将形状细化为单像素宽的中轴线（medial axis），保留拓扑结构。

**用途**：指纹识别、字符识别、管道网络分析

```python
import numpy as np
import cv2
from skimage.morphology import skeletonize, thin

# skimage 方法（推荐）
binary = (cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)[1]).astype(bool)
skeleton = skeletonize(binary)  # 返回 bool 数组
skeleton_img = (skeleton * 255).astype(np.uint8)

# OpenCV 迭代细化（Zhang-Suen 算法）
def zhang_suen_thinning(img_binary):
    """迭代细化直到稳定"""
    return cv2.ximgproc.thinning(img_binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

# 比较效果
img_thin_gs = zhang_suen_thinning(binary_img)  # 更快
```

---

## 粒度分析（Granulometry）

粒度分析通过对不同尺寸的结构元素依次进行开运算，测量图像中各种尺寸颗粒的分布。

**用途**：材料科学（颗粒粒径分布）、医学图像（细胞大小统计）、工业检测

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

def granulometry(binary_img, se_type=cv2.MORPH_ELLIPSE, max_radius=30):
    """
    计算粒度谱（Pattern Spectrum）
    返回每个尺寸下开运算后前景像素减少量
    """
    sizes = range(1, max_radius + 1)
    area = binary_img.sum() / 255

    spectrum = []
    prev_area = area
    for r in sizes:
        se = cv2.getStructuringElement(se_type, (2*r+1, 2*r+1))
        opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, se)
        curr_area = opened.sum() / 255
        spectrum.append(prev_area - curr_area)  # 被消除的像素数
        prev_area = curr_area

    return list(sizes), spectrum

# 可视化粒度谱
sizes, spectrum = granulometry(binary_img, max_radius=25)
plt.bar(sizes, spectrum)
plt.xlabel("结构元素半径（像素）")
plt.ylabel("消除的前景像素数")
plt.title("粒度谱（颗粒大小分布）")
plt.show()

# 峰值处对应图像中最常见的颗粒大小
dominant_size = sizes[np.argmax(spectrum)]
print(f"主要颗粒半径约为 {dominant_size} 像素")
```

---

## 连通域分析

形态学处理后，通常用连通域分析提取各个目标区域：

```python
import cv2
import numpy as np

def analyze_components(binary_img, min_area=100):
    """提取并过滤连通域"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )

    result = []
    for i in range(1, num_labels):  # 0 是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        result.append({
            "id": i, "area": area,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy)
        })
    return result

# 使用示例：工业检测中过滤小噪点，只保留大目标
components = analyze_components(binary_img, min_area=500)
print(f"检测到 {len(components)} 个目标")
```

---

## 参考资料

- 《数字图像处理》Gonzalez & Woods（形态学章节）
- OpenCV 文档：Morphological Transformations
- scikit-image 文档：skimage.morphology
