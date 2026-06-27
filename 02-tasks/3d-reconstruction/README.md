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
