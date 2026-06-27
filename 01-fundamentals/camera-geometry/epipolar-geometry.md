# 对极几何

## 核心概念

- **对极约束**：两张图像中同一 3D 点的对应关系满足 `x'^T F x = 0`
- **基础矩阵 F**（3×3，秩2）：包含内参信息
- **本质矩阵 E**（3×3，秩2）：纯旋转平移，适用于已标定相机 `E = K'^T F K`
- **极线**：一张图的点对应另一张图的一条线（而非一个点）

## 基础矩阵求解

```python
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
```

## 本质矩阵与位姿恢复

```python
E, mask = cv2.findEssentialMat(pts1, pts2, K)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
```

## 应用

- 双目匹配：极线约束将 2D 搜索降为 1D
- SfM / SLAM：初始化相机位姿
- 图像矫正（Stereo Rectification）

---

## 极线几何完整推导

### 几何关系

设两个已标定相机 C₁ 和 C₂，同一 3D 点 P 在两张图像中的投影分别为 x₁ 和 x₂。

```
          P
         /|
        / |
       /  |
      x₁  x₂
      |   |
      C₁  C₂
      
e₁（极点）= C₂ 在图像 1 中的投影
e₂（极点）= C₁ 在图像 2 中的投影
极线 l₁ = 过 x₁ 和 e₁ 的直线
极线 l₂ = 过 x₂ 和 e₂ 的直线
```

### 本质矩阵 E 的推导

已标定相机，归一化坐标为 `x̃ = K⁻¹ x`（齐次坐标）。

两相机坐标系的变换关系：`P₂ = R P₁ + t`

对极约束：向量 `t`、`P₁` 和 `R P₁ - t` 三者共面：

```
(R P₁ + t - t) · [t × (R P₁)] = 0
P₁ · [t × (R P₁)] = 0
P₁ · (t × R P₁) = 0         （混合积）
x̃₂^T (t× R) x̃₁ = 0          （代入投影）
x̃₂^T E x̃₁ = 0                （E = t× R）
```

其中 `t×` 是 t 的反对称矩阵（叉积矩阵）：

```
    [  0  -tz   ty]
t× =[  tz   0  -tx]
    [ -ty   tx   0]
```

```python
import numpy as np

def skew_symmetric(t):
    """构造平移向量的反对称矩阵"""
    tx, ty, tz = t
    return np.array([[  0, -tz,  ty],
                     [ tz,   0, -tx],
                     [-ty,  tx,   0]])

def essential_from_Rt(R, t):
    """由旋转矩阵和平移向量计算本质矩阵"""
    t_normalized = t / np.linalg.norm(t)  # 归一化（E 只有 5 自由度）
    return skew_symmetric(t_normalized) @ R

# 验证：E 的性质 - 两个相等的非零奇异值，一个零奇异值
R = np.eye(3)
t = np.array([1.0, 0.0, 0.0])
E = essential_from_Rt(R, t)
U, s, Vt = np.linalg.svd(E)
print("E 的奇异值:", s)  # → 应为 [1, 1, ~0]
```

### 基础矩阵 F 的推导

基础矩阵将内参信息编码进去，适用于未标定相机：

```
F = K₂⁻ᵀ E K₁⁻¹ = K₂⁻ᵀ (t× R) K₁⁻¹
```

像素坐标的对极约束：`x₂^T F x₁ = 0`

**F 的自由度**：3×3 矩阵有 9 个参数，减去尺度（1）和秩约束（det=0），共 7 个自由度，**8 点法**最少需要 8 对对应点（线性求解）。

```python
import cv2
import numpy as np

def demo_fundamental_matrix(img1, img2):
    """演示基础矩阵求解和极线绘制"""
    # 特征匹配
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # 求基础矩阵（RANSAC 鲁棒估计）
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    inlier_pts1 = pts1[mask.ravel() == 1]
    inlier_pts2 = pts2[mask.ravel() == 1]

    # 计算并绘制极线
    def draw_epilines(img1, img2, pts1, pts2, F):
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        h, w = img2.shape[:2]
        img2_draw = img2.copy()
        for line, pt in zip(lines2[:10], pts2[:10]):
            a, b, c = line
            # 极线：ax + by + c = 0
            x0, y0 = 0, int(-c/b)
            x1, y1 = w, int(-(c + a*w)/b)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(img2_draw, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img2_draw, tuple(pt.astype(int)), 5, color, -1)
        return img2_draw

    img2_epilines = draw_epilines(img1, img2, inlier_pts1, inlier_pts2, F)
    return F, img2_epilines
```

---

## 单应矩阵 H

### 定义

当两张图像之间的场景**是平面**，或**是纯旋转**（相机位置不变）时，两幅图像之间存在单应矩阵 H（3×3 满秩）：

```
x₂ = H x₁     （齐次坐标）
```

H 有 8 个自由度（3×3 矩阵减去尺度），**直接线性变换（DLT）** 最少需要 4 对对应点。

### 求解与应用

```python
import cv2
import numpy as np

def image_registration(img_src, img_dst, pts_src, pts_dst):
    """
    图像配准：用单应矩阵将源图像变换到目标图像坐标系
    用途：全景拼接、文档扫描矫正、AR 标记跟踪
    """
    # 求单应矩阵（RANSAC 鲁棒）
    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
    print("单应矩阵 H:\n", H)
    print(f"内点数: {mask.sum()}/{len(pts_src)}")

    # 透视变换（将源图像变换到目标坐标系）
    h, w = img_dst.shape[:2]
    warped = cv2.warpPerspective(img_src, H, (w, h))

    return H, warped

def perspective_correction(img, corners):
    """
    文档/棋盘透视矫正
    corners: 文档四个角点的像素坐标 [[tl, tr, br, bl]]
    """
    tl, tr, br, bl = corners

    # 计算目标矩形宽高
    width  = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    src_pts = np.float32([tl, tr, br, bl])

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, H, (width, height))
    return corrected
```

### H 与 F 的关系

| | 基础矩阵 F | 单应矩阵 H |
|---|---|---|
| 尺寸 | 3×3 | 3×3 |
| 秩 | 2 | 3 |
| 自由度 | 7 | 8 |
| 最少点数 | 8（线性）/ 7（非线性）| 4 |
| 适用场景 | 一般三维场景 | 平面场景或纯旋转 |
| 约束 | `x'^T F x = 0` | `x' = Hx` |

---

## RANSAC 鲁棒估计

### 原理

RANSAC（Random Sample Consensus）通过随机采样小子集估计模型，再统计内点（inliers）数量来找到最优解，能容忍大量错误匹配（外点）。

```
重复 N 次：
  1. 随机采样 m 个点（m=最小所需点数，如 8 点法取 8 点）
  2. 用这 m 个点计算模型参数（如 F 矩阵）
  3. 对所有点评估模型误差，满足 threshold 的为内点
  4. 若内点数 > 当前最优，保存该模型

迭代次数 N = log(1-p) / log(1-(1-e)^m)
    p: 至少一次采样全为内点的概率（通常取 0.99）
    e: 外点比例
    m: 每次采样点数
```

```python
import numpy as np
import cv2

def ransac_fundamental(pts1, pts2, threshold=3.0, confidence=0.99):
    """
    RANSAC 鲁棒估计基础矩阵，并可视化内外点
    """
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,  # 像素误差阈值
        confidence=confidence
    )

    inlier_mask = mask.ravel() == 1
    n_inliers = inlier_mask.sum()
    n_outliers = (~inlier_mask).sum()
    inlier_ratio = n_inliers / len(pts1)

    print(f"内点: {n_inliers}, 外点: {n_outliers}, 内点率: {inlier_ratio:.1%}")

    # 估计迭代次数（事后验证）
    e = 1 - inlier_ratio
    if e < 1:
        import math
        N = math.log(1 - confidence) / math.log(1 - (1-e)**8)
        print(f"理论上 RANSAC 只需 {N:.0f} 次迭代")

    return F, inlier_mask

# 也可用于单应矩阵
def ransac_homography(pts1, pts2, threshold=5.0):
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)
    return H, mask.ravel() == 1
```

---

## 参考资料

- 《Multiple View Geometry in Computer Vision》Hartley & Zisserman（对极几何圣经）
- OpenCV 文档：Camera Calibration and 3D Reconstruction
- 《视觉 SLAM 十四讲》第 7 章：视觉里程计
- Fischler & Bolles (1981)：Random Sample Consensus
