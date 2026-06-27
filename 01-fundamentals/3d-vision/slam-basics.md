# SLAM 基础

> Simultaneous Localization and Mapping：同时定位与建图

## 核心问题

- **定位**：我在哪？（估计相机/机器人位姿）
- **建图**：环境是什么样的？（构建地图）
- **闭环检测**：识别回到已经走过的地方，修正累积误差

## SLAM 框架组成

```
传感器输入 → 前端里程计 → 后端优化 → 闭环检测 → 建图
```

### 前端（Visual Odometry）
- 特征点法：ORB-SLAM3
- 直接法：LSD-SLAM、DSO
- 半直接法：SVO

### 后端优化
- 图优化（g2o、GTSAM）：位姿图、BA（Bundle Adjustment）
- 滤波法：EKF、UKF（轻量，精度稍低）

## 主流开源系统

| 系统 | 传感器 | 特点 |
|------|--------|------|
| ORB-SLAM3 | 单目/双目/RGB-D/IMU | 最完整，学习必读 |
| RTAB-Map | RGB-D/双目/LiDAR | 实用，闭环强 |
| LIO-SAM | LiDAR + IMU | 自动驾驶常用 |
| VINS-Mono | 单目 + IMU | 轻量，无人机 |

---

## SLAM 问题的数学定义

### 状态估计

SLAM 本质上是一个**最大后验估计（MAP）**问题：

```
x* = argmax P(x_{0:T}, m | z_{1:T}, u_{1:T})

其中：
x_{0:T} ：机器人位姿序列（待估计）
m        ：地图（路标点坐标，待估计）
z_{1:T} ：传感器观测序列（已知）
u_{1:T} ：控制输入（已知）
```

### 概率图模型

SLAM 对应一个**动态贝叶斯网络**，包含两类约束：

1. **运动模型**：`P(x_t | x_{t-1}, u_t)` — 上一位姿 + 控制输入预测当前位姿
2. **观测模型**：`P(z_t | x_t, m)` — 当前位姿下对地图的观测概率

**图优化视角**：将位姿和路标作为节点，约束（运动/观测）作为边，最小化所有约束误差之和：

```
E(x, m) = Σ_t ||x_t - f(x_{t-1}, u_t)||² + Σ_{t,i} ||z_{t,i} - h(x_t, m_i)||²
```

---

## ORB-SLAM3 安装与运行

```bash
# 依赖安装（Ubuntu 20.04）
sudo apt-get install libopencv-dev libeigen3-dev libpangolin-dev

# 克隆并编译
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3
chmod +x build.sh
./build.sh

# 运行 TUM 数据集（RGB-D 模式）
./Examples/RGB-D/rgbd_tum \
    Vocabulary/ORBvoc.txt \
    Examples/RGB-D/TUM1.yaml \
    /path/to/TUMdataset/rgbd_dataset_freiburg1_xyz \
    /path/to/TUMdataset/rgbd_dataset_freiburg1_xyz/associations.txt

# 运行 EuRoC 数据集（双目+IMU 模式）
./Examples/Stereo-Inertial/stereo_inertial_euroc \
    Vocabulary/ORBvoc.txt \
    Examples/Stereo-Inertial/EuRoC.yaml \
    /path/to/MH01 \
    Examples/Stereo-Inertial/EuRoC_TimeStamps/MH01.txt \
    dataset-MH01_stereoi

# 保存轨迹（KeyFrameTrajectory.txt 格式：timestamp tx ty tz qx qy qz qw）
```

---

## 视觉里程计数学推导（从本质矩阵到位姿）

**特征点匹配 → 相对位姿恢复**的核心步骤：

### 1. 计算本质矩阵（Essential Matrix）

对极约束：`x2^T · E · x1 = 0`

其中 x1、x2 为归一化图像坐标（去畸变后），E = [t]×R

使用**8点法**或 RANSAC + 5点法求解 E：

```python
import cv2

# 特征匹配
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

# 相机内参
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# 计算本质矩阵（RANSAC 去除外点）
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
```

### 2. 从 E 分解得到 R 和 t

```python
# SVD 分解：E = U·Σ·V^T
# 有 4 种 R/t 组合，需通过三角化验证正确解
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
# R: (3,3) 旋转矩阵，t: (3,1) 平移向量（单位向量，尺度未知）
print(f"R:\n{R}\nt: {t.flatten()}")
```

### 3. 三角化恢复 3D 点

```python
# 构建投影矩阵
P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
P2 = K @ np.hstack([R, t])

# 三角化
pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)  # (4, N) 齐次坐标
pts3d = (pts4d[:3] / pts4d[3]).T  # (N, 3)
```

---

## 回环检测：词袋模型（BoW）原理

BoW 用于快速判断当前帧是否到达之前访问过的场景。

**核心步骤**：

1. **离线训练词汇表**：
   - 从大量图像中提取 ORB/BRIEF 描述子
   - 用 K-means 层次聚类构建**词汇树**（Vocabulary Tree）
   - 树的叶子节点即为"视觉词"（Visual Words）

2. **在线图像表示**：
   - 提取当前帧的描述子
   - 遍历词汇树，将每个描述子映射到最近的视觉词
   - 用 TF-IDF 加权统计词频，得到**词袋向量（Bag of Words Vector）**

3. **相似度计算**：
   - 比较当前帧与历史关键帧的 BoW 向量相似度（L1 距离）
   - 超过阈值则触发回环候选
   - 进一步用几何验证（PnP + RANSAC）确认回环

```bash
# DBoW2 库（ORB-SLAM3 使用的 BoW 实现）
# 预训练词汇表：ORB_SLAM3/Vocabulary/ORBvoc.txt
# 包含 10 万个视觉词，查找 O(log N) 复杂度
```

---

## 位姿图结构说明（图优化）

SLAM 后端的图优化本质：**最小化所有约束的加权误差平方和**。

**位姿图（Pose Graph）**：

```
节点：每个关键帧的位姿 T_i ∈ SE(3)（6DOF：旋转+平移）
边：
  ├── 里程计边：相邻帧之间的相对变换（前端估计）
  └── 回环边：回环检测到的相对变换（通常精度更高）

优化目标：
  E = Σ_{(i,j)∈E} || log(T_{ij}^{-1} · T_i^{-1} · T_j) ||²_Σ_{ij}
```

**Bundle Adjustment（BA）**：同时优化相机位姿和 3D 路标点，是 SLAM 精度最高的方式，但计算量大（Full BA）。ORB-SLAM3 使用局部 BA（只优化当前窗口内的关键帧）平衡精度与速度。

---

## SLAM 评估工具（evo 工具包）

```bash
# 安装 evo
pip install evo --upgrade --no-binary evo

# 评估绝对轨迹误差（ATE）
evo_ape tum \
    groundtruth.txt \
    estimated_trajectory.txt \
    --align --correct_scale \
    --plot --save_results results/

# 评估相对位姿误差（RPE）
evo_rpe tum \
    groundtruth.txt \
    estimated_trajectory.txt \
    --delta 1 --delta_unit f \
    --align --plot

# 轨迹可视化
evo_traj tum \
    estimated_trajectory.txt \
    --ref groundtruth.txt \
    --plot --plot_mode xy

# 对比多个算法
evo_traj tum \
    orb_slam3.txt \
    vins_mono.txt \
    --ref groundtruth.txt \
    --plot

# 评估指标说明
# ATE（绝对轨迹误差）：整体漂移，衡量建图精度
# RPE（相对位姿误差）：局部精度，衡量里程计精度
# 输出：RMSE、mean、median、std、min、max
```

**轨迹文件格式（TUM）**：

```
# timestamp tx ty tz qx qy qz qw
1305031102.175304 1.3405 0.6266 1.6575 0.6571 0.6399 -0.2938 -0.2781
1305031102.211214 1.3373 0.6274 1.6576 0.6574 0.6397 -0.2937 -0.2781
```
