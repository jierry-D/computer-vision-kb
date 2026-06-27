# 传统特征提取

## 特征描述子对比

| 特征 | 尺度不变 | 旋转不变 | 速度 | 用途 |
|------|---------|---------|------|------|
| SIFT | ✅ | ✅ | 慢 | 图像匹配、拼接 |
| SURF | ✅ | ✅ | 中 | SIFT 加速版 |
| ORB | ✅ | ✅ | 快 | 实时 SLAM、AR |
| BRIEF | ❌ | ❌ | 极快 | 需配合检测器 |
| HOG | ❌ | 部分 | 中 | 行人检测 |
| LBP | ❌ | ✅ | 快 | 纹理、人脸识别 |

## OpenCV 示例（ORB）

```python
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(img, None)

# BF 匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
```

## 注意

- SIFT/SURF 曾有专利限制，OpenCV 4.x 已开放
- 深度学习特征（SuperPoint、DISK）已逐步替代传统特征

---

## SIFT 完整算法步骤

SIFT（Scale-Invariant Feature Transform，尺度不变特征变换）由 David Lowe 于 2004 年提出，是经典中的经典。

### 步骤一：构建高斯尺度空间（Gaussian Scale Space）

对图像用不同 σ 的高斯核卷积，构建多尺度表示：

```
L(x, y, σ) = G(x, y, σ) * I(x, y)
```

图像组织为多个 **Octave（组）**，每组内有多层，组间分辨率减半。

### 步骤二：DOG（Difference of Gaussian）极值检测

相邻尺度的高斯图像相减，近似 LoG，计算高效：

```
D(x, y, σ) = L(x, y, kσ) - L(x, y, σ)
```

在 3D 空间（x, y, σ）中寻找局部极值点（候选关键点）：每个点与同尺度 8 邻居 + 上下层各 9 邻居共 26 个点比较。

### 步骤三：关键点精化与过滤

- **亚像素位置精化**：用泰勒展开拟合，调整到更精确位置
- **去除低对比度点**：DOG 响应太小的点丢弃（阈值通常 0.03）
- **去除边缘响应**：用 Hessian 矩阵的特征值比检查（类似 Harris），去除边缘上的虚假极值

### 步骤四：方向分配（旋转不变性来源）

在关键点邻域内统计梯度方向直方图（36个bin，每bin10°），取主方向为特征方向。若存在超过主峰 80% 的次峰，该点生成多个关键点（不同方向）。

### 步骤五：构建 SIFT 描述子（128维）

以关键点为中心，在 4×4 个子区域内各统计 8 个方向的梯度直方图，共 4×4×8 = 128 维向量，并做归一化（抵抗光照变化）。

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def demo_sift(img1_path, img2_path):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    # ─── 创建 SIFT 检测器 ───
    sift = cv2.SIFT_create(
        nfeatures=500,       # 最多检测多少个关键点
        nOctaveLayers=3,     # 每组的层数
        contrastThreshold=0.04,  # 低对比度过滤阈值
        edgeThreshold=10,    # 边缘过滤阈值
        sigma=1.6            # 初始高斯σ
    )

    # ─── 检测并计算描述子 ───
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    print(f"图1: {len(kp1)} 个关键点, 图2: {len(kp2)} 个关键点")
    print(f"描述子维度: {des1.shape}")  # (N, 128)

    # ─── 关键点可视化 ───
    img1_kp = cv2.drawKeypoints(
        img1, kp1, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # 显示方向和尺度
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(img1_kp, cmap='gray')
    plt.title(f"SIFT 关键点（共{len(kp1)}个），圆圈大小=尺度，方向=箭头")
    plt.axis('off')
    plt.show()

    return kp1, kp2, des1, des2
```

---

## FLANN 快速匹配

FLANN（Fast Library for Approximate Nearest Neighbors）使用 KD-Tree 或 LSH 等结构加速最近邻搜索，比暴力匹配（BFMatcher）快数倍。

```python
import cv2
import numpy as np

def sift_flann_match(img1_path, img2_path, ratio_thresh=0.75):
    """
    SIFT 特征提取 + FLANN 匹配 + Lowe's ratio test 过滤
    """
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # ─── FLANN 参数配置 ───
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # 越大越精确但越慢
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # ─── KNN 匹配（找最近的2个） ───
    matches_knn = flann.knnMatch(des1, des2, k=2)

    # ─── Lowe's Ratio Test 过滤错误匹配 ───
    # 若最近距离 < ratio × 次近距离，认为是好的匹配
    good_matches = [m for m, n in matches_knn if m.distance < ratio_thresh * n.distance]
    print(f"总候选匹配: {len(matches_knn)}, 过滤后: {len(good_matches)}")

    # ─── 可视化匹配结果 ───
    img_matches = cv2.drawMatchesKnn(
        img1, kp1, img2, kp2,
        [[m] for m in good_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # ─── RANSAC 估计单应矩阵（进一步过滤外点） ───
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = [m for m, flag in zip(good_matches, mask.ravel()) if flag]
        print(f"RANSAC 内点: {len(inliers)} / {len(good_matches)}")
        return H, inliers

    return None, good_matches

# ORB + FLANN（适用于二进制描述子）
def orb_flann_match(img1, img2):
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # ORB 用 LSH 索引（二进制描述子）
    FLANN_INDEX_LSH = 6
    index_params  = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                         key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if len([m,n])==2 and m.distance < 0.75 * n.distance]
    return kp1, kp2, good
```

---

## HOG（方向梯度直方图）特征详解

HOG 是行人检测领域的经典特征，被用于 DPM（Deformable Part Model）等方法。

```python
import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure

def compute_hog(img, visualize=True):
    """
    计算 HOG 特征
    """
    # skimage 实现（更灵活）
    fd, hog_image = hog(
        img,
        orientations=9,       # 梯度方向数
        pixels_per_cell=(8, 8),   # 每个 cell 的大小
        cells_per_block=(2, 2),   # 每个 block 包含的 cell 数
        visualize=True,
        channel_axis=-1 if len(img.shape)==3 else None
    )
    print(f"HOG 特征向量长度: {len(fd)}")

    if visualize:
        hog_img_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        return fd, hog_img_rescaled
    return fd

# OpenCV HOGDescriptor（用于行人检测）
def pedestrian_detection(img):
    hog_detector = cv2.HOGDescriptor()
    hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # detectMultiScale: 多尺度检测
    boxes, weights = hog_detector.detectMultiScale(
        img, winStride=(4, 4), padding=(8, 8), scale=1.05
    )
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img, boxes
```

---

## 参考资料

- SIFT 原始论文：Lowe, D.G. (2004). Distinctive Image Features from Scale-Invariant Keypoints
- HOG 原始论文：Dalal & Triggs (2005). Histograms of Oriented Gradients for Human Detection
- OpenCV 文档：Feature Detection and Description
- SuperPoint（深度学习特征）：https://arxiv.org/abs/1712.07629
