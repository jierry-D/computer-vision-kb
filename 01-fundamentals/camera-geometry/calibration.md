# 相机标定

## 张正友标定法（最常用）

**原理**：使用多张不同角度的棋盘格图像，求解内参和畸变系数。

### 流程

1. 打印棋盘格，固定在平面上
2. 从多个角度拍摄（建议 15-20 张，覆盖视野各区域）
3. 检测角点：`cv2.findChessboardCorners`
4. 亚像素精化：`cv2.cornerSubPix`
5. 求解标定：`cv2.calibrateCamera`

### OpenCV 示例

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,   # 3D 角点坐标列表
    imgpoints,   # 2D 图像角点列表
    img_size,
    None, None
)
# mtx: 内参矩阵, dist: 畸变系数
```

### 去畸变

```python
undistorted = cv2.undistort(img, mtx, dist, None, mtx)
```

## 评估指标

- **重投影误差（RMS）**：< 0.5 pixel 为优，< 1.0 pixel 可用

---

## 完整单相机标定代码

```python
import cv2
import numpy as np
import glob
import os

def calibrate_camera(image_dir, pattern_size=(9, 6), square_size=0.025):
    """
    完整单相机标定流程
    pattern_size: 棋盘格内角点数 (列数-1, 行数-1)
    square_size: 方格物理尺寸（米）
    """
    # 准备 3D 角点坐标（Z=0 平面）
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size  # 转换为实际物理尺寸（米）

    objpoints = []  # 3D 坐标列表
    imgpoints = []  # 2D 坐标列表

    images = glob.glob(os.path.join(image_dir, "*.jpg"))
    img_size = None
    success_count = 0

    for fname in images:
        img  = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  # (W, H)

        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            print(f"未检测到角点: {fname}")
            continue

        # 亚像素精化（提高精度）
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners_refined)
        success_count += 1

        # 可视化（调试用）
        cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)

    print(f"成功使用 {success_count}/{len(images)} 张图像")

    # 执行标定
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    print(f"重投影误差 RMS: {rms:.4f} 像素")
    print("内参矩阵 K:\n", mtx)
    print("畸变系数 dist:\n", dist)

    return mtx, dist, rvecs, tvecs, rms
```

---

## 标定质量评估

### 1. 重投影误差（最重要指标）

将标定后的 3D 角点重新投影到图像，计算与检测到的 2D 角点之间的距离。

```python
def evaluate_calibration(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    """计算每张图像的重投影误差"""
    total_error = 0
    per_image_errors = []

    for i, (objp, imgp) in enumerate(zip(objpoints, imgpoints)):
        projected, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgp, projected, cv2.NORM_L2) / len(projected)
        per_image_errors.append(error)
        total_error += error

    mean_error = total_error / len(objpoints)
    print(f"平均重投影误差: {mean_error:.4f} 像素")

    # 找出误差最大的图像（可能需要剔除）
    worst_idx = np.argmax(per_image_errors)
    print(f"误差最大的图像: 第 {worst_idx} 张，误差 = {per_image_errors[worst_idx]:.4f}")

    return mean_error, per_image_errors
```

### 2. 误差评判标准

| RMS 重投影误差 | 评价 | 建议 |
|--------------|------|------|
| < 0.3 px | 优秀 | 可用于高精度测量 |
| 0.3 ~ 0.5 px | 良好 | 工业/科研通用 |
| 0.5 ~ 1.0 px | 可接受 | 一般应用 |
| > 1.0 px | 较差 | 重新采集数据 |

### 3. 检查畸变系数合理性

```python
def validate_distortion(dist):
    """检查畸变系数是否在合理范围"""
    k1, k2, p1, p2 = dist[0, 0], dist[0, 1], dist[0, 2], dist[0, 3]
    k3 = dist[0, 4] if dist.shape[1] > 4 else 0

    print(f"k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}")
    print(f"p1={p1:.4f}, p2={p2:.4f}")

    # 异常检测
    if abs(k1) > 1.0:
        print("⚠️ k1 偏大，可能标定有误")
    if abs(p1) > 0.1 or abs(p2) > 0.1:
        print("⚠️ 切向畸变较大，检查镜头安装是否平行")
```

---

## 标定数据采集技巧

### 角度分布建议

良好的标定图像应覆盖相机整个视野，角度分布均匀：

```
✅ 推荐的角度分布（15-20 张）：
  - 正面中心 × 2 张（验证用）
  - 向左/右倾斜 15°~30° 各 3 张
  - 向上/下倾斜 15°~30° 各 3 张
  - 四个角落各 1~2 张（覆盖边缘畸变）
  - 带绕光轴旋转的斜角 2~3 张

❌ 避免：
  - 所有图像从同一方向拍摄
  - 棋盘格只出现在图像中心
  - 角度变化过小（< 10°）
```

### 光照建议

- 使用均匀漫射光，避免强烈反光（导致角点检测失败）
- 棋盘格纸张避免皱褶（平面度要求 < 0.1mm）
- 打印分辨率 ≥ 300dpi，黑白对比度清晰

### 棋盘格尺寸选择

```python
# 棋盘格方格大小建议：占图像 1/4 ~ 1/3 的宽度
# 例如 1920×1080 图像，建议方格 40~80mm，内角点 9×6

# 检测时的参数调整
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
ret, corners = cv2.findChessboardCorners(gray, (9, 6), flags=flags)
```

---

## 多相机联合标定（双目/多目）

以双目相机为例，在单目标定基础上进行联合标定，求解两相机间的相对位姿 (R, T)。

```python
import cv2
import numpy as np

def stereo_calibrate(objpoints, imgpoints_l, imgpoints_r,
                     K_l, dist_l, K_r, dist_r, img_size):
    """
    双目相机联合标定
    输入：左右相机各自的标定结果（或可同时优化）
    输出：相对旋转 R 和平移 T（单位：标定板方格物理尺寸）
    """
    flags = cv2.CALIB_FIX_INTRINSIC  # 固定已知内参，只求外参
    # 若需同时优化内参，去掉 CALIB_FIX_INTRINSIC

    rms, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        K_l, dist_l,
        K_r, dist_r,
        img_size,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    )

    print(f"双目标定 RMS: {rms:.4f}")
    print("相对旋转 R:\n", R)
    print("相对平移 T:", T.ravel(), "（基线长度:", np.linalg.norm(T), "m）")
    print("本质矩阵 E:\n", E)
    print("基础矩阵 F:\n", F)

    return R, T, E, F

# 双目矫正（使对应点位于同一水平线）
def stereo_rectify(K_l, dist_l, K_r, dist_r, img_size, R, T):
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        K_l, dist_l, K_r, dist_r, img_size, R, T,
        alpha=0  # 0=裁剪黑边, 1=保留所有
    )
    # Q 矩阵用于深度图计算（视差 → 3D 坐标）
    return R1, R2, P1, P2, Q
```

---

## MATLAB 与 OpenCV 标定工具箱对比

| 特性 | MATLAB Camera Calibration Toolbox | OpenCV |
|------|----------------------------------|--------|
| 易用性 | 高（GUI 交互） | 中（需编写代码） |
| 精度 | 高 | 高（与 MATLAB 相当） |
| 自动化 | 较差 | 好（适合批量处理） |
| 鱼眼支持 | 有（独立 fisheye toolbox） | 有（cv2.fisheye） |
| 多相机支持 | 有（Stereo Camera Calibrator） | 有（stereoCalibrate） |
| 输出格式 | .mat | numpy 数组，可保存为 yaml/json |
| 价格 | 商业（需授权） | 开源免费 |

### 保存和加载标定结果

```python
import cv2
import numpy as np

def save_calibration(fname, K, dist, img_size):
    """保存标定参数到 YAML 文件"""
    fs = cv2.FileStorage(fname, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.write("image_width",  img_size[0])
    fs.write("image_height", img_size[1])
    fs.release()
    print(f"标定参数已保存到 {fname}")

def load_calibration(fname):
    """从 YAML 文件加载标定参数"""
    fs = cv2.FileStorage(fname, cv2.FILE_STORAGE_READ)
    K    = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    w    = int(fs.getNode("image_width").real())
    h    = int(fs.getNode("image_height").real())
    fs.release()
    return K, dist, (w, h)

# 使用示例
save_calibration("calibration.yaml", mtx, dist, (640, 480))
K, dist, size = load_calibration("calibration.yaml")
```

---

## 参考资料

- 张正友标定法论文：Zhang (2000). A Flexible New Technique for Camera Calibration
- OpenCV 文档：Camera Calibration
- MATLAB Camera Calibration Toolbox（Bouguet）
- 《视觉 SLAM 十四讲》第 5 章：相机与图像
