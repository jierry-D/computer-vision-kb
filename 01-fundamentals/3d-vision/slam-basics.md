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
