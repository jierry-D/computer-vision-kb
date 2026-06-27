# 对极几何

## 核心概念

- **对极约束**：两张图像中同一 3D 点的对应关系满足 `x'^T F x = 0`
- **基础矩阵 F**（3×3，秩2）：包含内参信息
- **本质矩阵 E**（3×3，秩2）：纯旋转平移，适用于已标定相机 `E = K'^T F K`
- **极线**：一张图的点对应另一张图的一条线（而非一个点）

## 基础矩阵求解

```python
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
```

## 本质矩阵与位姿恢复

```python
E, mask = cv2.findEssentialMat(pts1, pts2, K)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
```

## 应用

- 双目匹配：极线约束将 2D 搜索降为 1D
- SfM / SLAM：初始化相机位姿
- 图像矫正（Stereo Rectification）
