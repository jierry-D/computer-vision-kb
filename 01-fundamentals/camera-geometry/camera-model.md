# 相机模型

## 针孔相机模型

```
3D 世界坐标 → 相机坐标 → 图像平面坐标 → 像素坐标
     [R|t]           K（内参矩阵）
```

### 内参矩阵 K

```
K = [fx  0  cx]
    [ 0 fy  cy]
    [ 0  0   1]
```

- `fx`, `fy`：焦距（像素单位）
- `cx`, `cy`：主点（光轴与像平面的交点）

### 外参 [R | t]

- R：3×3 旋转矩阵，相机相对世界坐标系的朝向
- t：3×1 平移向量

## 畸变模型

| 类型 | 系数 | 常见镜头 |
|------|------|---------| 
| 径向畸变 | k1, k2, k3 | 普通镜头（桶形/枕形） |
| 切向畸变 | p1, p2 | 镜头安装不平行 |
| 鱼眼畸变 | 独立模型 | 超广角鱼眼镜头 |

## 参考

- OpenCV 文档：Camera Calibration and 3D Reconstruction

---

## 完整坐标变换推导

### 四个坐标系

1. **世界坐标系** (X_w, Y_w, Z_w)：任意定义的参考系
2. **相机坐标系** (X_c, Y_c, Z_c)：以光心为原点，Z 轴指向前方
3. **图像坐标系** (x, y)：单位为毫米，以主点为原点
4. **像素坐标系** (u, v)：单位为像素，以左上角为原点

### 世界坐标 → 相机坐标（外参变换）

```
[X_c]   [R | t] [X_w]
[Y_c] =         [Y_w]
[Z_c]           [Z_w]
                [ 1 ]
```

R 是 3×3 旋转矩阵，t 是 3×1 平移向量。[R|t] 构成 3×4 外参矩阵。

### 相机坐标 → 图像坐标（透视投影）

```
x = f × X_c / Z_c
y = f × Y_c / Z_c
```

这一步将三维点投影到焦平面，Z_c 的除法是透视效果（近大远小）的来源。

### 图像坐标 → 像素坐标（内参变换）

```
u = fx × x + cx
v = fy × y + cy
```

- `fx = f / dx`（f 为物理焦距，dx 为像素宽度，单位 mm/pixel）
- `fy = f / dy`（dy 为像素高度）

### 完整投影矩阵（齐次坐标）

```
[u]       [fx   0  cx  0] [R | t] [X_w]
[v] = (1/Z_c) × [ 0  fy  cy  0]         [Y_w]
[1]       [ 0   0   1  0]         [Z_w]
                                         [ 1 ]
          ↑____内参矩阵 K___↑  ↑外参↑
```

简写为：`s × p = K × [R|t] × P_w`（s 为深度缩放因子）

```python
import numpy as np

def project_points(P_world, K, R, t):
    """
    将世界坐标点投影到像素坐标
    P_world: (N, 3) 世界坐标
    K: (3, 3) 内参矩阵
    R: (3, 3) 旋转矩阵
    t: (3,)  平移向量
    返回: (N, 2) 像素坐标
    """
    # 世界 → 相机
    P_cam = (R @ P_world.T).T + t  # (N, 3)

    # 透视除法（相机 → 归一化图像坐标）
    P_norm = P_cam[:, :2] / P_cam[:, 2:3]  # (N, 2)

    # 归一化图像坐标 → 像素
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * P_norm[:, 0] + cx
    v = fy * P_norm[:, 1] + cy
    return np.stack([u, v], axis=-1)

# 示例
K = np.array([[800,   0, 320],
              [  0, 800, 240],
              [  0,   0,   1]], dtype=float)
R = np.eye(3)
t = np.array([0, 0, 5.0])  # 相机在 Z=5 处看向原点

points_3d = np.array([[0.0, 0.0, 0.0],  # 原点
                       [1.0, 0.0, 0.0],  # X 轴上 1m 处
                       [0.0, 1.0, 0.0]]) # Y 轴上 1m 处

pixels = project_points(points_3d, K, R, t)
print("投影像素坐标:\n", pixels)
```

---

## 内参矩阵各参数物理意义

| 参数 | 物理含义 | 典型值（640×480 相机） |
|------|---------|----------------------|
| fx | 水平方向焦距（像素） | 500~1000 px |
| fy | 垂直方向焦距（像素） | 500~1000 px（通常 ≈ fx） |
| cx | 主点 x 坐标 | ≈ 图像宽度/2 = 320 px |
| cy | 主点 y 坐标 | ≈ 图像高度/2 = 240 px |
| s  | 像素斜切（skew），现代相机≈0 | 0 |

**水平视场角（FOV）**：

```
FOV_h = 2 × arctan(W / (2 × fx))
FOV_v = 2 × arctan(H / (2 × fy))
```

```python
import numpy as np

def fov_from_intrinsics(K, width, height):
    fx, fy = K[0, 0], K[1, 1]
    fov_h = 2 * np.degrees(np.arctan(width  / (2 * fx)))
    fov_v = 2 * np.degrees(np.arctan(height / (2 * fy)))
    return fov_h, fov_v

K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
print(fov_from_intrinsics(K, 640, 480))  # → (43.6°, 33.4°)
```

---

## 畸变校正公式

### 径向畸变（桶形/枕形）

```
x' = x (1 + k1 r² + k2 r⁴ + k3 r⁶)
y' = y (1 + k1 r² + k2 r⁴ + k3 r⁶)
```

其中 `r² = x² + y²`，(x, y) 是归一化图像坐标。

- k1 > 0：桶形畸变（图像边缘向外弯曲）
- k1 < 0：枕形畸变（图像边缘向内弯曲）

### 切向畸变

```
x' = x + [2 p1 xy + p2 (r² + 2x²)]
y' = y + [p1 (r² + 2y²) + 2 p2 xy]
```

### OpenCV 畸变校正

```python
import cv2
import numpy as np

# 假设已标定得到内参和畸变系数
K    = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
dist = np.array([[-0.3, 0.1, 0.001, 0.002, 0.0]])  # [k1,k2,p1,p2,k3]

img = cv2.imread("distorted.jpg")

# 方法1：直接去畸变（每次重新计算映射）
undistorted = cv2.undistort(img, K, dist)

# 方法2：预先计算映射表（批量处理时更快）
h, w = img.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_16SC2)
undistorted_fast = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
# alpha=0: 去畸变后裁掉黑边; alpha=1: 保留所有像素（含黑边）
```

---

## 鱼眼相机模型（Kannala-Brandt）

### 普通针孔模型的局限

针孔模型在视场角超过 ~120° 时失效（投影点趋于无穷）。鱼眼相机视场角可达 180°~220°，需要专用模型。

### Kannala-Brandt 等距投影模型

将入射角 θ 直接映射到径向距离 r：

```
r(θ) = k1 θ + k2 θ³ + k3 θ⁵ + k4 θ⁷ + k5 θ⁹
```

其中 `θ = arctan(√(X²+Y²) / Z)` 是光线与光轴的夹角。

```python
import cv2
import numpy as np

# OpenCV 鱼眼标定（使用 fisheye 模块）
objpoints = []  # 3D 角点坐标
imgpoints = []  # 2D 图像角点

# ... 同普通标定，收集棋盘格角点 ...

# 鱼眼标定
K_fish = np.zeros((3, 3))
D_fish = np.zeros((4, 1))   # 鱼眼畸变系数 [k1,k2,k3,k4]
flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
rms, K_fish, D_fish, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints, imgpoints, img.shape[:2][::-1],
    K_fish, D_fish, flags=flags
)

# 鱼眼去畸变
img_undist = cv2.fisheye.undistortImage(img, K_fish, D_fish, Knew=K_fish)

# 鱼眼等距投影转透视投影
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K_fish, D_fish, np.eye(3), K_fish, img.shape[:2][::-1], cv2.CV_16SC2
)
undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
```

---

## 参考资料

- OpenCV 文档：Camera Calibration and 3D Reconstruction
- Kannala & Brandt (2006)：A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses
- 《视觉 SLAM 十四讲》高翔（相机模型章节）
- 《Multiple View Geometry in Computer Vision》Hartley & Zisserman
