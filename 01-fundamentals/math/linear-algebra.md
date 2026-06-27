# 线性代数

## 核心概念

- **向量与矩阵**：表示图像坐标、变换、特征
- **矩阵乘法**：仿射变换、卷积的矩阵形式
- **特征值与特征向量**：PCA 降维、协方差矩阵分析
- **SVD（奇异值分解）**：图像压缩、低秩近似、最小二乘

## 在机器视觉中的应用

| 概念 | 应用场景 |
|------|---------|
| 旋转矩阵 | 相机位姿、坐标系变换 |
| 单应矩阵 H | 图像对齐、透视变换 |
| SVD | 基础矩阵求解、PnP |
| 特征值分解 | Harris 角点检测 |

---

## 矩阵运算示例（NumPy）

```python
import numpy as np

# 创建矩阵
A = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[5, 6], [7, 8]], dtype=float)

# 矩阵乘法
C = A @ B          # 推荐写法
C = np.dot(A, B)   # 等价

# 转置
AT = A.T

# 逆矩阵
A_inv = np.linalg.inv(A)
print("A @ A_inv =\n", A @ A_inv)  # 应为单位矩阵

# 行列式
det_A = np.linalg.det(A)

# 范数
norm_A = np.linalg.norm(A, ord='fro')  # Frobenius 范数
norm_v = np.linalg.norm(np.array([3, 4]))  # L2 范数 = 5.0

# 解线性方程 Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)
```

---

## 特征值与特征向量

### 定义

对于方阵 A，若存在非零向量 v 满足：

```
A v = λ v
```

则 λ 称为特征值，v 称为对应的特征向量。

### 推导步骤

1. 由 `Av = λv` 变形为 `(A - λI)v = 0`
2. 要有非零解，需要 `det(A - λI) = 0`（特征多项式）
3. 求解特征多项式得到所有特征值 λ₁, λ₂, ...
4. 将每个 λᵢ 代回 `(A - λᵢI)v = 0` 求对应特征向量

### NumPy 求解

```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(A)
print("特征值:", eigenvalues)        # [5. 2.]
print("特征向量（列向量）:\n", eigenvectors)

# 验证：A @ v = λ * v
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v   = eigenvectors[:, i]
    print(f"λ={lam:.1f}: A@v = {A @ v},  λ*v = {lam * v}")

# 对称矩阵的特征分解（实际应用更常见）
cov = A.T @ A  # 协方差矩阵是对称半正定矩阵
vals, vecs = np.linalg.eigh(cov)  # eigh 专用于对称矩阵，结果更稳定
```

### 在计算机视觉中的意义

- **PCA 降维**：协方差矩阵的特征向量方向为主成分方向，特征值大小表示方差
- **Harris 角点**：角点响应由梯度矩阵的最小特征值决定
- **图像配准**：本质矩阵 E 的奇异值分解用于恢复位姿

---

## SVD（奇异值分解）

### 定义

任意 m×n 矩阵 A 均可分解为：

```
A = U Σ V^T
```

- U（m×m）：左奇异向量矩阵，列向量正交
- Σ（m×n）：对角矩阵，对角元素 σ₁ ≥ σ₂ ≥ ... ≥ 0 为奇异值
- V（n×n）：右奇异向量矩阵，列向量正交

### NumPy 计算

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

U, s, Vt = np.linalg.svd(A, full_matrices=True)
print("奇异值:", s)  # [1.684e+01, 1.068e+00, ~0]

# 重建矩阵
Sigma = np.zeros_like(A)
np.fill_diagonal(Sigma, s)
A_reconstructed = U @ Sigma @ Vt
print("重建误差:", np.linalg.norm(A - A_reconstructed))
```

### SVD 实现图像压缩

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 读取灰度图像
img = np.array(Image.open("image.jpg").convert("L"), dtype=float)

U, s, Vt = np.linalg.svd(img, full_matrices=False)

def compress(U, s, Vt, k):
    """保留前 k 个奇异值重建图像"""
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# 对比不同压缩率
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, k in zip(axes, [5, 20, 50, len(s)]):
    compressed = np.clip(compress(U, s, Vt, k), 0, 255)
    ratio = k * (img.shape[0] + img.shape[1] + 1) / img.size
    ax.imshow(compressed, cmap='gray')
    ax.set_title(f"k={k}\n压缩率={ratio:.2%}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# 奇异值能量分布
total_energy = np.sum(s**2)
cumulative   = np.cumsum(s**2) / total_energy
k_90 = np.searchsorted(cumulative, 0.90) + 1
print(f"保留 90% 能量只需 {k_90} 个奇异值（共 {len(s)} 个）")
```

---

## 旋转矩阵的三种表示

三维空间中的旋转可以用多种方式等价表示，各有优缺点。

### 1. 旋转矩阵（SO(3)）

3×3 正交矩阵，满足 `R^T R = I`，`det(R) = 1`。

```
绕 Z 轴旋转 θ：
Rz = [cosθ  -sinθ  0]
     [sinθ   cosθ  0]
     [0      0     1]
```

**优点**：直接用于坐标变换；**缺点**：9个参数，存在冗余，不利于插值。

### 2. 欧拉角（Euler Angles）

将旋转分解为绕三个轴的连续旋转，常用 **Roll-Pitch-Yaw**（RPY）：

```
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
```

**缺点**：存在"万向锁"（Gimbal Lock）问题，绕 Y 轴旋转 ±90° 时损失一个自由度。

```python
import numpy as np

def euler_to_rotation(roll, pitch, yaw):
    """RPY 欧拉角 → 旋转矩阵"""
    Rx = np.array([[1,           0,            0],
                   [0,  np.cos(roll), -np.sin(roll)],
                   [0,  np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1,             0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]])
    return Rz @ Ry @ Rx
```

### 3. 四元数（Quaternion）

用 4 个参数 `q = (qw, qx, qy, qz)`，`|q| = 1` 表示旋转。

```
q = cos(θ/2) + sin(θ/2)(ix + jy + kz)
```

**优点**：避免万向锁，插值平滑（Slerp），数值稳定；**缺点**：直觉理解难。

```python
# 使用 scipy 操作四元数
from scipy.spatial.transform import Rotation as R

# 从欧拉角创建四元数
r = R.from_euler('zyx', [45, 30, 15], degrees=True)
quat = r.as_quat()          # [qx, qy, qz, qw]
rot_mat = r.as_matrix()     # 转回旋转矩阵

# 四元数插值（Slerp）
from scipy.spatial.transform import Slerp
key_rots   = R.from_euler('z', [0, 90], degrees=True)
key_times  = [0, 1]
slerp      = Slerp(key_times, key_rots)
interp_rot = slerp([0.25, 0.5, 0.75])  # 插值中间帧
```

### 4. 轴角（Axis-Angle）

用旋转轴 n（单位向量）和旋转角 θ 表示：`(n, θ)` 或 Rodrigues 向量 `r = θ·n`（3个参数）。

```python
import cv2
import numpy as np

# 旋转矩阵 ↔ Rodrigues 向量（OpenCV常用）
rvec = np.array([0.1, 0.2, 0.3])          # Rodrigues 向量
R_mat, _ = cv2.Rodrigues(rvec)            # → 旋转矩阵
rvec_back, _ = cv2.Rodrigues(R_mat)       # → 旋转向量
```

### 三种表示对比

| 表示方式 | 参数个数 | 万向锁 | 插值 | 计算旋转 | 典型用途 |
|---------|---------|--------|------|---------|---------|
| 旋转矩阵 | 9（实际3自由度）| 无 | 不直接 | 直接相乘 | 坐标变换 |
| 欧拉角 | 3 | 有 | 不平滑 | 需转换 | 人机交互、可视化 |
| 四元数 | 4 | 无 | Slerp 平滑 | 乘法合成 | 动画、SLAM |
| 轴角 | 3 | 无 | 线性近似 | 需转换 | 标定、优化 |

---

## 参考资料

- 《线性代数》吉尔伯特·斯特朗（MIT 18.06）
- 3Blue1Brown「线性代数的本质」系列
- 《计算机视觉：算法与应用》Richard Szeliski（第2版）
- SciPy Rotation 文档：https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
