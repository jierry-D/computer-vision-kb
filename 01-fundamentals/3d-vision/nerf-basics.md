# NeRF 与 3DGS

## NeRF（Neural Radiance Field）

**核心思想**：用 MLP 隐式表达 3D 场景，输入 (x,y,z,θ,φ) 输出颜色+密度，通过体渲染合成新视角。

### 流程
1. 采集多角度图像 + 相机位姿（COLMAP 重建）
2. 训练 MLP 拟合场景
3. 体渲染（Volume Rendering）合成任意视角

### 主要变体

| 版本 | 改进 |
|------|------|
| NeRF（原版）| 基础，慢 |
| Instant-NGP | 哈希编码加速，分钟级训练 |
| Mip-NeRF 360 | 处理无界场景 |
| Nerfacto（nerfstudio）| 工程实用版 |

## 3D Gaussian Splatting（3DGS）

**核心思想**：用数百万个 3D 高斯球显式表达场景，光栅化渲染，训练和渲染比 NeRF 快数十倍。

- 训练：~30 分钟（vs NeRF 数小时）
- 渲染：实时（>100 FPS）
- 缺点：存储占用大（数百MB ~ GB）

## 工具

- nerfstudio：`pip install nerfstudio`，统一框架
- gaussian-splatting 官方代码：CUDA 实现
