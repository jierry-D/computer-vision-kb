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

---

## 体渲染方程数学推导

NeRF 的核心是**体积渲染（Volume Rendering）**，将光线在 3D 空间中的透射/吸收过程建模为积分。

### 连续积分形式

给定从相机原点 o 出发，方向为 d 的光线 r(t) = o + t·d：

```
C(r) = ∫_{t_n}^{t_f} T(t) · σ(r(t)) · c(r(t), d) dt

其中：
T(t) = exp(-∫_{t_n}^{t} σ(r(s)) ds)  （透射率：光线到达 t 处未被吸收的概率）
σ(x) ：体密度（volume density），表示光子在该点被吸收的概率
c(x,d)：辐射颜色（RGB），依赖位置和方向
```

### 离散化采样（实际计算）

将光线离散为 N 个采样点：

```
C(r) ≈ Σ_{i=1}^{N} T_i · (1 - exp(-σ_i · δ_i)) · c_i

T_i = exp(-Σ_{j=1}^{i-1} σ_j · δ_j)   （前 i-1 个点的累积透射率）
δ_i = t_{i+1} - t_i                     （相邻采样点间距）
```

**分层采样（Hierarchical Sampling）**：
1. **粗采样（Coarse）**：均匀采样 64 个点，用粗网络预测密度
2. **细采样（Fine）**：根据粗网络预测的密度分布，重采样 128 个点（密度大的区域更密集）

```python
import torch

def volume_render(rgb, sigma, t_vals):
    """
    rgb:   (B, N, 3) 各采样点颜色
    sigma: (B, N)    各采样点体密度
    t_vals:(B, N)    各采样点深度值
    """
    # 计算相邻采样间距 δ
    deltas = t_vals[..., 1:] - t_vals[..., :-1]
    # 最后一段设为无穷大（防止截断）
    deltas = torch.cat([deltas, torch.full_like(deltas[..., :1], 1e10)], dim=-1)

    # 计算 alpha（每点被吸收的概率）
    alpha = 1.0 - torch.exp(-sigma * deltas)

    # 计算透射率 T_i（前缀乘积）
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1)[..., :-1]

    # 加权求和得到最终颜色
    weights = transmittance * alpha  # (B, N)
    rgb_map = (weights.unsqueeze(-1) * rgb).sum(dim=-2)  # (B, 3)
    depth_map = (weights * t_vals).sum(dim=-1)            # (B,)
    return rgb_map, depth_map, weights
```

---

## Instant-NGP 哈希编码原理

Instant-NGP 用**多分辨率哈希编码**替代 NeRF 中昂贵的位置编码（Positional Encoding），将训练速度从数小时压缩到数分钟。

**核心思想**：

1. 将 3D 空间划分为 L 个不同分辨率的 3D 网格（从粗到细）
2. 每个网格的角点被哈希映射到一个大小为 T 的特征表中
3. 对于任意输入位置 x，在每个分辨率层：
   - 找到 8 个邻近网格角点
   - 通过哈希函数获取各角点特征
   - 三线性插值得到当前分辨率的特征向量
4. 将所有分辨率的特征向量**拼接**后送入轻量 MLP

```
哈希函数：h(x) = (x1·π1 ⊕ x2·π2 ⊕ x3·π3) mod T
其中 π1=1, π2=2654435761, π3=805459861（质数），⊕ 为异或
T：哈希表大小（通常 2^{14}~2^{24}）
```

**优势**：
- 特征表直接可学习（梯度反传到哈希表）
- 哈希冲突产生隐式正则化效果
- 多分辨率捕捉不同尺度的细节

---

## nerfstudio 完整使用流程

```bash
# 1. 安装
pip install nerfstudio

# 2. 数据准备：用 COLMAP 从图像序列重建相机位姿
ns-process-data images \
    --data /path/to/images \
    --output-dir /path/to/processed_data

# 如果是视频，先提取帧
ns-process-data video \
    --data video.mp4 \
    --output-dir /path/to/processed_data \
    --num-frames-target 300

# 3. 训练（选择不同 NeRF 方法）
# Nerfacto（推荐，速度快精度高）
ns-train nerfacto \
    --data /path/to/processed_data \
    --output-dir ./outputs \
    --max-num-iterations 30000

# Instant-NGP（更快）
ns-train instant-ngp \
    --data /path/to/processed_data

# 3DGS（最快渲染）
ns-train splatfacto \
    --data /path/to/processed_data

# 4. 渲染新视角（交互式浏览器查看）
ns-viewer --load-config outputs/.../config.yml

# 5. 导出视频/图像
ns-render camera-path \
    --load-config outputs/.../config.yml \
    --camera-path-filename camera_path.json \
    --output-path renders/output.mp4

# 6. 导出 mesh（实验性）
ns-export marching-cubes \
    --load-config outputs/.../config.yml \
    --output-dir exports/mesh
```

---

## 3DGS Splatting 渲染过程详解

3D Gaussian Splatting 使用**显式高斯原语**表达场景，通过光栅化而非光线追踪渲染，速度极快。

**每个 3D 高斯球的参数**：
- 位置 μ ∈ R³（中心坐标）
- 协方差矩阵 Σ ∈ R^{3×3}（形状/方向/尺寸，分解为旋转 q + 缩放 s）
- 颜色 c（球谐函数 SH 系数，支持视角相关颜色）
- 不透明度 α ∈ [0,1]

**渲染步骤**：

```
1. 视锥剔除（View Frustum Culling）
   丢弃相机视野外的高斯球

2. 3D → 2D 投影（Splatting）
   将 3D 高斯投影为 2D 椭圆：
   Σ' = J · W · Σ · W^T · J^T
   其中 J 为雅可比矩阵，W 为视图变换矩阵

3. 深度排序（按相机深度由远到近）
   对所有覆盖该像素的高斯按深度排序（α-compositing 需要）

4. Alpha 混合（前向到后向）
   对每个像素 p，从前到后累积：
   C(p) = Σ_i c_i · α_i' · Π_{j<i}(1 - α_j')
   其中 α_i' = α_i · exp(-0.5 · (p-μ_i')^T · Σ_i'^{-1} · (p-μ_i'))
```

**自适应密度控制**：训练过程中动态**增加/删除**高斯球：
- 梯度大且不透明度高 → 克隆/分裂（增加细节）
- 不透明度接近 0 → 删除（减少冗余）
- 高斯球过大 → 分裂

---

## NeRF vs 3DGS vs MVS 详细对比

| 对比维度 | NeRF（原版）| Instant-NGP | 3DGS | MVS/SfM（COLMAP）|
|----------|------------|-------------|------|-----------------|
| **表达形式** | 隐式（MLP）| 隐式（哈希+MLP）| 显式（高斯球）| 显式（稀疏/稠密点云）|
| **训练时间** | 1~2 天 | 5~10 分钟 | 30~60 分钟 | 数分钟（特征匹配）|
| **渲染速度** | 慢（分钟/帧）| ~60 FPS | >100 FPS（实时）| 光栅化即时 |
| **渲染质量** | 高 | 中高 | 高 | 中（依赖稠密重建）|
| **内存占用** | 低（MLP 参数小）| 中（哈希表）| 高（GB 级）| 中 |
| **编辑性** | 差（需重训）| 差 | 中（可操作高斯球）| 好（直接编辑点云）|
| **无纹理区域** | 较好 | 中 | 差（梯度无法驱动）| 差（匹配失败）|
| **动态场景** | 需特殊处理 | 需特殊处理 | D-3DGS 扩展 | 不支持 |
| **新视角合成** | 优秀 | 优秀 | 优秀 | 一般 |
| **代表工具** | nerf-pytorch | tiny-cuda-nn | 官方 CUDA 实现 | COLMAP |

---

## 常见失败 Case 和解决方法

| 问题现象 | 可能原因 | 解决方法 |
|----------|----------|----------|
| 训练 Loss 不收敛 | 相机位姿错误 | 重新运行 COLMAP，检查 transforms.json |
| 渲染结果有云雾/浮尘 | 密度场过于弥散 | 增大 near plane，调小初始 sigma |
| 场景边界渲染错误 | 无界场景未处理 | 使用 Mip-NeRF 360 或 nerfstudio 的 nerfacto |
| 高光/镜面区域糊 | NeRF 假设漫反射 | 使用 NeRF-W 或 Ref-NeRF 处理高光 |
| 3DGS 出现针状伪影 | 高斯球过于细长 | 调低 opacity reset 间隔，增大正则化 |
| 3DGS 内存溢出 | 高斯球数量过多 | 降低 densification 阈值，增大剔除频率 |
| nerfstudio 训练极慢 | 未使用 GPU 或 tiny-cuda-nn 未安装 | 确认 CUDA 版本，重新安装 `pip install nerfstudio[dev]` |
| COLMAP 重建失败 | 图像重叠不足/运动模糊 | 增大重叠率（>60%），减少运动，增加图像数量 |

**数据采集建议**：
- 图像重叠度 > 60%，避免快速移动
- 避免纯白墙、镜面等无纹理区域
- 多角度围绕物体拍摄（而非单方向），确保完整覆盖
- 光线均匀稳定，避免强逆光
