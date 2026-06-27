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

---

## 点云文件格式说明

| 格式 | 特点 | 常用场景 |
|------|------|----------|
| `.pcd` | PCL 原生格式，支持 ASCII/二进制，含字段定义头部 | 机器人/PCL 生态 |
| `.ply` | 通用多边形格式，支持点云+网格，可存颜色法向量 | Open3D/Blender |
| `.bin` | KITTI 数据集格式，原始二进制，每点 [x,y,z,intensity] float32 | 自动驾驶数据集 |
| `.las/.laz` | 航测/GIS 标准格式，支持大规模地理坐标点云 | 激光扫描/GIS |
| `.xyz` | 简单文本格式，每行一个点，空格分隔 | 调试/可视化 |

```python
import numpy as np

# 读取 KITTI .bin 格式
def load_kitti_bin(bin_path):
    """返回 (N, 4) 数组：[x, y, z, intensity]"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

# 保存为 .pcd（ASCII）
def save_pcd_ascii(points, filename):
    """points: (N, 3) numpy array"""
    n = len(points)
    with open(filename, 'w') as f:
        f.write(f"# .PCD v0.7 - Point Cloud Data\n")
        f.write(f"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
        f.write(f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA ascii\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
```

---

## Open3D 完整操作代码

```python
import open3d as o3d
import numpy as np

# 1. 读取点云
pcd = o3d.io.read_point_cloud("cloud.pcd")
print(f"点云包含 {len(pcd.points)} 个点")

# 2. 体素下采样（降低密度，加速处理）
down_pcd = pcd.voxel_down_sample(voxel_size=0.05)
print(f"下采样后 {len(down_pcd.points)} 个点")

# 3. 法向量估计（用于配准/重建）
down_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
# 将法向量方向统一朝向相机
down_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

# 4. 统计滤波（去除离群点）
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
clean_pcd = pcd.select_by_index(ind)

# 5. ICP 点云配准（将 source 配准到 target）
source = o3d.io.read_point_cloud("source.pcd")
target = o3d.io.read_point_cloud("target.pcd")
source.estimate_normals()
target.estimate_normals()

# 粗配准：FPFH 特征 + RANSAC
def fpfh_ransac_registration(source, target, voxel_size=0.05):
    radius_feature = voxel_size * 5
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, src_fpfh, tgt_fpfh, mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3, checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

# 精配准：ICP
def icp_refine(source, target, init_transform, threshold=0.02):
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return result.transformation

# 6. 可视化（多个点云，不同颜色）
source_colored = o3d.io.read_point_cloud("source.pcd")
target_colored = o3d.io.read_point_cloud("target.pcd")
source_colored.paint_uniform_color([1, 0, 0])  # 红色
target_colored.paint_uniform_color([0, 1, 0])  # 绿色
o3d.visualization.draw_geometries([source_colored, target_colored])
```

---

## PointNet 关键代码结构

PointNet 的核心设计：
1. **对称函数（Max Pooling）**：使输出对点的排列不变（permutation invariant）
2. **T-Net（变换网络）**：学习输入空间和特征空间的对齐变换矩阵

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """输入/特征变换网络：预测 k×k 变换矩阵"""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn = nn.ModuleList([nn.BatchNorm1d(c) for c in [64, 128, 1024, 512, 256]])

    def forward(self, x):
        # x: (B, k, N)
        x = F.relu(self.bn[0](self.conv1(x)))
        x = F.relu(self.bn[1](self.conv2(x)))
        x = F.relu(self.bn[2](self.conv3(x)))
        x = x.max(dim=2)[0]  # 对称函数：全局最大池化 → (B, 1024)
        x = F.relu(self.bn[3](self.fc1(x)))
        x = F.relu(self.bn[4](self.fc2(x)))
        x = self.fc3(x)  # (B, k*k)
        # 初始化为单位矩阵
        iden = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(x.size(0), 1)
        x = x + iden
        return x.view(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    """PointNet 特征提取器"""
    def __init__(self, global_feat=True):
        super().__init__()
        self.tnet3 = TNet(k=3)     # 输入变换
        self.tnet64 = TNet(k=64)   # 特征变换
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        # x: (B, 3, N)
        n_pts = x.size(2)

        # 输入变换
        trans = self.tnet3(x)
        x = torch.bmm(trans, x)  # (B, 3, N)

        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)

        # 特征变换
        trans_feat = self.tnet64(x)
        x = torch.bmm(trans_feat, x)  # (B, 64, N)
        point_feat = x  # 保留逐点特征（分割任务用）

        x = F.relu(self.bn2(self.conv2(x)))   # (B, 128, N)
        x = self.bn3(self.conv3(x))            # (B, 1024, N)

        # 关键：对称聚合函数（Max Pooling）
        x = x.max(dim=2)[0]  # (B, 1024) — 全局特征，与点的顺序无关

        if self.global_feat:
            return x  # 分类任务
        else:
            # 分割任务：全局特征 + 逐点特征拼接
            x = x.unsqueeze(2).repeat(1, 1, n_pts)
            return torch.cat([x, point_feat], dim=1)  # (B, 1088, N)
```

---

## 点云分割和目标检测流程

### 3D 语义分割（PointNet++ 风格）

```
输入点云 (N, 3+C)
    ↓ Set Abstraction（分层采样 + 局部特征提取）
    ↓ Feature Propagation（插值传播回密集点）
    ↓ 逐点分类头（MLP + Softmax）
输出：每点的类别标签 (N, num_classes)
```

### LiDAR 3D 目标检测流程（PointPillars）

```
1. 点云 → 柱状体素化（Pillarization）
   将空间划分为垂直柱体（Pillar），每柱内提取点云特征

2. Pillar Feature Network（PFN）
   对每个 Pillar 内的点用 PointNet 提取特征 → 2D 伪图像

3. Backbone 2D CNN + FPN
   在伪图像上做 2D 目标检测（类似 SSD）

4. 检测头：输出 3D 框参数 [x,y,z,l,w,h,θ]
```

---

## 点云与深度图的互转代码

```python
import numpy as np

def depth_to_pointcloud(depth_map, fx, fy, cx, cy, depth_scale=1000.0):
    """
    深度图 → 点云
    depth_map: (H, W) uint16 或 float，单位为毫米（除以 depth_scale 得米）
    fx,fy,cx,cy: 相机内参
    返回：(N, 3) float32 点云
    """
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_map.astype(np.float32) / depth_scale  # 转换为米
    mask = z > 0  # 过滤无效深度
    x = (u[mask] - cx) * z[mask] / fx
    y = (v[mask] - cy) * z[mask] / fy
    return np.stack([x, y, z[mask]], axis=1)


def pointcloud_to_depth(points, fx, fy, cx, cy, H, W):
    """
    点云 → 深度图（稀疏）
    points: (N, 3) 相机坐标系下的点
    """
    depth_map = np.zeros((H, W), dtype=np.float32)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = z > 0
    u = (x[mask] * fx / z[mask] + cx).astype(int)
    v = (y[mask] * fy / z[mask] + cy).astype(int)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    depth_map[v[valid], u[valid]] = z[mask][valid]
    return depth_map


# 示例：RealSense 内参
fx, fy, cx, cy = 615.0, 615.0, 320.0, 240.0
depth = np.random.randint(500, 3000, (480, 640), dtype=np.uint16)  # 模拟深度图
pcd_points = depth_to_pointcloud(depth, fx, fy, cx, cy)
print(f"生成点云：{pcd_points.shape}")  # (N, 3)
```
