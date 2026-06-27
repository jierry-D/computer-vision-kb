# 双目视觉

## 原理

通过两个相机的视差（disparity）反推深度：

```
depth Z = (f × B) / disparity
```

- `f`：焦距（像素）
- `B`：基线（两相机光心距离）
- `disparity`：同一点在左右图像的像素偏移量

## 流程

1. **双目标定**：分别标定左右相机，再联合标定得到 R, T
2. **图像矫正（Rectification）**：使对应点在同一水平线上
3. **视差计算**：SGBM / BM 算法，或深度学习方法
4. **深度图生成**：由视差图换算

## OpenCV 示例

```python
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=9)
disparity = stereo.compute(img_left_gray, img_right_gray)
depth = (focal_length * baseline) / disparity
```

## 深度学习方法

- PSMNet、RAFT-Stereo、CREStereo

---

## 双目标定完整流程

### Step 1：分别标定左右相机，再联合标定

```python
import cv2
import numpy as np
import glob

def full_stereo_calibration(left_dir, right_dir,
                             pattern_size=(9, 6), square_size=0.025):
    """
    完整双目标定流程
    """
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints_l, imgpoints_r = [], []

    left_imgs  = sorted(glob.glob(f"{left_dir}/*.jpg"))
    right_imgs = sorted(glob.glob(f"{right_dir}/*.jpg"))
    img_size   = None

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for lf, rf in zip(left_imgs, right_imgs):
        gray_l = cv2.cvtColor(cv2.imread(lf), cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(cv2.imread(rf), cv2.COLOR_BGR2GRAY)
        img_size = gray_l.shape[::-1]

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size)

        if ret_l and ret_r:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)

    # 先单独标定左右相机（获得初始内参）
    _, K_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None)
    _, K_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None)

    # 联合标定（固定内参，优化外参 R, T）
    rms, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K_l, dist_l, K_r, dist_r,
        img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    )

    print(f"双目标定 RMS: {rms:.4f} 像素")
    baseline = np.linalg.norm(T)
    print(f"基线长度: {baseline*100:.2f} cm")

    return K_l, dist_l, K_r, dist_r, R, T, img_size
```

---

## 图像矫正（Stereo Rectification）

矫正后，左右图像中的对应点位于同一水平扫描线，视差计算简化为一维搜索。

```python
import cv2
import numpy as np

def stereo_rectify_and_remap(K_l, dist_l, K_r, dist_r, R, T, img_size):
    """
    双目矫正：计算映射表
    """
    # 计算矫正变换
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        K_l, dist_l, K_r, dist_r,
        img_size, R, T,
        alpha=0,    # 0=裁剪黑边，1=保留所有像素
        flags=cv2.CALIB_ZERO_DISPARITY  # 使左右图像水平对齐
    )

    # 生成映射表（只需计算一次）
    map_l1, map_l2 = cv2.initUndistortRectifyMap(
        K_l, dist_l, R1, P1, img_size, cv2.CV_16SC2
    )
    map_r1, map_r2 = cv2.initUndistortRectifyMap(
        K_r, dist_r, R2, P2, img_size, cv2.CV_16SC2
    )

    print("Q 矩阵（视差 → 3D 坐标）:\n", Q)
    # Q 矩阵含义：
    # Q = [1  0    0      -cx_l  ]
    #     [0  1    0      -cy    ]
    #     [0  0    0       fx    ]
    #     [0  0  -1/B  (cx_l-cx_r)/B]

    return map_l1, map_l2, map_r1, map_r2, Q

def rectify_images(img_l, img_r, map_l1, map_l2, map_r1, map_r2):
    """对图像对应用矫正映射"""
    rect_l = cv2.remap(img_l, map_l1, map_l2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map_r1, map_r2, cv2.INTER_LINEAR)

    # 验证矫正效果：绘制水平线，观察对应点是否在同一行
    h, w = rect_l.shape[:2]
    combined = np.hstack([rect_l, rect_r])
    for y in range(0, h, 50):
        cv2.line(combined, (0, y), (2*w, y), (0, 255, 0), 1)

    return rect_l, rect_r, combined
```

---

## 视差计算与可视化

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_disparity(rect_l_gray, rect_r_gray, method='sgbm'):
    """
    计算视差图
    """
    if method == 'bm':
        # StereoBM：速度快，适合纹理丰富的场景
        stereo = cv2.StereoBM_create(
            numDisparities=128,  # 必须是 16 的倍数
            blockSize=15         # 奇数，越大越平滑
        )
    else:
        # StereoSGBM：准确，适合复杂场景
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8  * 3 * 5**2,   # 惩罚系数（平滑性）
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    disparity = stereo.compute(rect_l_gray, rect_r_gray).astype(np.float32) / 16.0
    # 注意：OpenCV 返回值 ×16，需除以 16 才是实际视差（像素）

    return disparity

def visualize_disparity(disparity, save_path=None):
    """视差图可视化"""
    # 过滤无效值（视差 < 0 的区域）
    disp_vis = disparity.copy()
    disp_vis[disp_vis < 0] = 0

    # 归一化到 0-255
    disp_norm = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 用伪彩色映射（更直观）
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.imshow(disp_norm, cmap='gray')
    plt.title('视差图（灰度）')
    plt.colorbar(label='视差（像素）')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB))
    plt.title('视差图（伪彩色）- 红=近，蓝=远')
    plt.colorbar()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return disp_color
```

---

## 深度图与点云生成

```python
import cv2
import numpy as np
import open3d as o3d

def disparity_to_depth(disparity, focal_length, baseline):
    """
    视差图 → 深度图
    depth = f * B / disparity
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.where(
            disparity > 0,
            focal_length * baseline / disparity,
            0  # 无效区域深度为 0
        )
    return depth

def depth_to_pointcloud_opencv(disparity, Q, img_color=None):
    """
    使用 OpenCV reprojectImageTo3D 将视差图转点云
    Q: stereoRectify 返回的 Q 矩阵
    """
    # 生成 3D 坐标
    points_3d = cv2.reprojectImageTo3D(disparity, Q)  # shape: (H, W, 3)

    # 过滤无效点
    mask = (disparity > disparity.min()) & (points_3d[:, :, 2] < 10.0)  # 去掉超远和无效点

    points = points_3d[mask]  # (N, 3)
    colors = img_color[mask] / 255.0 if img_color is not None else None

    return points, colors

def save_pointcloud(points, colors, output_path):
    """保存点云为 PLY 文件（使用 Open3D）"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors[:, ::-1])  # BGR→RGB
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云保存至 {output_path}，共 {len(points)} 个点")

def visualize_pointcloud(points, colors=None):
    """Open3D 交互式点云可视化"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd],
                                       window_name="双目点云",
                                       width=1280, height=720)

# 完整流程示例
def stereo_to_pointcloud(img_l, img_r, K_l, dist_l, K_r, dist_r, R, T):
    """端到端：双目图像对 → 点云"""
    img_size = img_l.shape[1::-1]

    # 矫正
    map_l1, map_l2, map_r1, map_r2, Q = stereo_rectify_and_remap(
        K_l, dist_l, K_r, dist_r, R, T, img_size
    )
    rect_l = cv2.remap(img_l, map_l1, map_l2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map_r1, map_r2, cv2.INTER_LINEAR)

    # 视差计算
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    disparity = compute_disparity(gray_l, gray_r, method='sgbm')

    # 转点云
    points, colors = depth_to_pointcloud_opencv(disparity, Q, img_color=rect_l)
    print(f"生成点云：{len(points)} 个点")

    return points, colors, disparity
```

---

## 双目视觉常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 视差图出现大量无效区（空洞）| 纹理缺失（白墙、天空） | 增加人工纹理，或用深度学习方法 |
| 边缘区域视差不准 | 遮挡（左相机可见，右相机看不到）| 左右一致性检查（disp12MaxDiff）|
| 视差图有竖向条纹 | 矫正效果不好 | 重新标定，检查重投影误差 |
| 近处/远处深度误差大 | 基线过短/过长 | 根据测量范围选择基线：`B ≈ 深度/10` |
| 计算速度慢 | SGBM 复杂度高 | 降低分辨率、numDisparities，或用GPU加速 |

---

## 参考资料

- OpenCV 文档：Camera Calibration and 3D Reconstruction（stereoCalibrate, stereoRectify）
- 《Multiple View Geometry in Computer Vision》Hartley & Zisserman
- PSMNet 论文：Chang & Chen (2018). Pyramid Stereo Matching Network
- RAFT-Stereo：Lipson et al. (2021). RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching
- Open3D 文档：http://www.open3d.org/docs/
