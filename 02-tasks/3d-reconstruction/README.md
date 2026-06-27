# 三维重建

## 方法分类

| 方法 | 输入 | 代表 |
|------|------|------|
| SfM | 多视角图像 | COLMAP |
| MVS | SfM 位姿 + 多视图 | OpenMVS、ACMH |
| SLAM | 视频流 | ORB-SLAM3、RTAB-Map |
| 深度融合 | RGB-D | KinectFusion、Open3D |
| NeRF | 多视角 + 位姿 | Instant-NGP、nerfstudio |
| 3DGS | 多视角 + 位姿 | gaussian-splatting |

## 典型流程（SfM + MVS）

```
图像采集 → COLMAP SfM（位姿估计+稀疏点云）
         → MVS 密集重建（稠密点云）
         → 网格重建（Poisson / TSDF）
         → 纹理贴图
```

## 工具链

| 工具 | 用途 |
|------|------|
| COLMAP | SfM 标准工具 |
| OpenMVS | MVS 密集重建 |
| Open3D | 点云/网格处理、可视化 |
| MeshLab | 网格编辑 |
| nerfstudio | NeRF/3DGS 统一框架 |

## 评估指标

- **Accuracy / Completeness / F-score**（DTU/Tanks and Temples）
- **PSNR / SSIM / LPIPS**（新视角合成质量）

---

## COLMAP 完整使用流程

### 稀疏重建（SfM）

```bash
# 假设图像在 ./images/ 目录下

# 1. 特征提取（SIFT）
colmap feature_extractor \
    --database_path ./database.db \
    --image_path ./images \
    --ImageReader.single_camera 1 \       # 所有图像同一相机
    --SiftExtraction.use_gpu 1

# 2. 特征匹配（穷举匹配，图像少时用）
colmap exhaustive_matcher \
    --database_path ./database.db \
    --SiftMatching.use_gpu 1

# 图像多时（>100张）用序列匹配更快
# colmap sequential_matcher --database_path ./database.db

# 3. 增量式稀疏重建（SfM）
mkdir -p ./sparse
colmap mapper \
    --database_path ./database.db \
    --image_path ./images \
    --output_path ./sparse \
    --Mapper.num_threads 8

# 4. 查看结果（稀疏点云 + 相机位姿）
colmap gui  # 可视化界面
# 或转换为文本格式
colmap model_converter \
    --input_path ./sparse/0 \
    --output_path ./sparse/0 \
    --output_type TXT
```

### 密集重建（MVS）

```bash
# 1. 对图像做立体矫正（undistortion）
mkdir -p ./dense
colmap image_undistorter \
    --image_path ./images \
    --input_path ./sparse/0 \
    --output_path ./dense \
    --output_type COLMAP

# 2. 多视图立体匹配（生成深度图）
colmap patch_match_stereo \
    --workspace_path ./dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.gpu_index 0

# 3. 深度图融合（生成稠密点云）
colmap stereo_fusion \
    --workspace_path ./dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path ./dense/fused.ply

# 4. 泊松网格重建（可选）
colmap poisson_mesher \
    --input_path ./dense/fused.ply \
    --output_path ./dense/meshed-poisson.ply

# 5. Delaunay 网格重建（可选，更快）
colmap delaunay_mesher \
    --input_path ./dense \
    --output_path ./dense/meshed-delaunay.ply
```

---

## SfM 核心步骤详解

### 特征匹配与几何验证

```python
# SfM 流程伪代码（理解原理）
# 1. 特征提取
keypoints, descriptors = sift.detect_and_compute(image)

# 2. 特征匹配（暴力匹配 / FLANN 近似最近邻）
matches = bf_matcher.knnMatch(desc1, desc2, k=2)
# Lowe's ratio test（去除模糊匹配）
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

# 3. 几何验证（基础矩阵 / 本质矩阵估计）
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
# RANSAC 估计本质矩阵（已知内参）
E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix,
                                method=cv2.RANSAC, prob=0.999, threshold=1.0)
# 分解本质矩阵得到 R, t
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix)

# 4. 三角化（已知两视图位姿，重建 3D 点）
proj1 = camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
proj2 = camera_matrix @ np.hstack([R, t])
points_4d = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
points_3d = (points_4d[:3] / points_4d[3]).T  # 齐次坐标转 3D

# 5. 增量注册新图像（PnP）
_, rvec, tvec, inliers = cv2.solvePnPRansac(
    obj_pts_3d, img_pts_2d, camera_matrix, None
)
```

---

## nerfstudio：从数据到渲染的完整命令

```bash
# 安装
pip install nerfstudio

# 1. 数据处理（自动运行 COLMAP + 格式转换）
ns-process-data images \
    --data ./images \
    --output-dir ./data/my_scene \
    --num-downscales 2   # 生成 1/2, 1/4 分辨率用于加速

# 2. 训练 NeRF（nerfacto，速度与质量均衡）
ns-train nerfacto \
    --data ./data/my_scene \
    --max-num-iterations 30000 \
    --viewer.start-train False  # 不开 web viewer（服务器训练用）

# 训练 3D Gaussian Splatting（速度更快，质量更高）
ns-train splatfacto \
    --data ./data/my_scene \
    --max-num-iterations 30000

# 3. 渲染新视角（沿预设相机路径）
ns-render camera-path \
    --load-config outputs/my_scene/nerfacto/config.yml \
    --camera-path-filename ./camera_path.json \
    --output-path ./renders/video.mp4

# 4. 导出点云/网格
ns-export pointcloud \
    --load-config outputs/my_scene/nerfacto/config.yml \
    --output-dir ./exports/pcd

ns-export tsdf \
    --load-config outputs/my_scene/nerfacto/config.yml \
    --output-dir ./exports/mesh
```

---

## Open3D 处理重建结果

### 点云滤波与可视化

```python
import open3d as o3d
import numpy as np

# 加载点云
pcd = o3d.io.read_point_cloud("fused.ply")
print(f"原始点数: {len(pcd.points)}")

# 1. 统计离群点滤波（去除孤立噪声点）
pcd_clean, ind = pcd.remove_statistical_outlier(
    nb_neighbors=20,    # 每个点考虑最近 20 个邻居
    std_ratio=2.0        # 超过 2 倍标准差视为离群点
)
print(f"滤波后点数: {len(pcd_clean.points)}")

# 2. 体素下采样（均匀稀疏化，加速后续处理）
pcd_down = pcd_clean.voxel_down_sample(voxel_size=0.01)

# 3. 法向量估计（网格重建前必须）
pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
# 统一法向量方向（朝向相机）
pcd_down.orient_normals_towards_camera_location()

# 4. 可视化
o3d.visualization.draw_geometries([pcd_down],
                                   point_show_normal=False)
```

### 泊松网格重建

```python
# 泊松重建（需要法向量）
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd_down,
    depth=9,          # 八叉树深度，越大越精细（建议 8-11）
    width=0,
    scale=1.1,
    linear_fit=False
)

# 去除低密度的"离散"三角面（边界噪声）
density_threshold = np.quantile(np.asarray(densities), 0.1)
vertices_to_remove = np.asarray(densities) < density_threshold
mesh.remove_vertices_by_mask(vertices_to_remove)

# 简化网格（减少面数，适合实时渲染）
mesh_simplified = mesh.simplify_quadric_decimation(
    target_number_of_triangles=100000
)
mesh_simplified.compute_vertex_normals()
o3d.io.write_triangle_mesh("output_mesh.ply", mesh_simplified)
```

### 纹理贴图

```python
# Open3D 纹理贴图（需要提供相机参数和原始图像）
# 使用 COLMAP 稠密重建的 dense 目录

mesh_with_tex = o3d.pipelines.color_map.run_rigid_optimizer(
    mesh_simplified,
    rgbd_images,         # [N] 个 RGBD 图像
    camera_trajectory,   # 相机位姿序列
    o3d.pipelines.color_map.RigidOptimizerOption(
        maximum_iteration=300
    )
)
o3d.io.write_triangle_mesh("textured_mesh.ply", mesh_with_tex)
```

**踩坑记录**：
- COLMAP 需要图像间有足够重叠（建议 60% 以上），否则无法匹配；无人机/环绕拍摄效果最好
- NeRF/3DGS 对相机位姿精度要求极高，COLMAP 的 mapper 失败是最常见的问题，可尝试 `--Mapper.ba_global_use_pba 0`
- 泊松重建的深度参数 depth=9 对应约 1000 万面，渲染前务必做简化
- Open3D 法向量方向估计不稳定时，可手动用 `orient_normals_consistent_tangent_plane` 替代
