# 点云处理

## 点云表示

- **无序点集**：N × (x, y, z) 或 N × (x, y, z, r, g, b, intensity)
- 来源：LiDAR、深度相机（RealSense/Kinect）、SfM/MVS 重建

## 经典网络

| 网络 | 年份 | 核心贡献 |
|------|------|---------|
| PointNet | 2017 | 直接处理无序点云，对称函数聚合 |
| PointNet++ | 2017 | 局部特征提取，多尺度分组 |
| DGCNN | 2019 | 动态图卷积 |
| VoxelNet | 2018 | 体素化后用 3D CNN |
| PointPillars | 2019 | 柱状体素，LiDAR 检测高效方案 |

## 常用操作

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("cloud.pcd")
# 下采样
down = pcd.voxel_down_sample(voxel_size=0.05)
# 法向量估计
pcd.estimate_normals()
# 可视化
o3d.visualization.draw_geometries([pcd])
```

## 应用场景

- 自动驾驶 3D 目标检测
- 机器人抓取位姿估计
- 三维重建与场景理解
