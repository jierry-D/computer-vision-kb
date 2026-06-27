# 相机标定

## 张正友标定法（最常用）

**原理**：使用多张不同角度的棋盘格图像，求解内参和畸变系数。

### 流程

1. 打印棋盘格，固定在平面上
2. 从多个角度拍摄（建议 15-20 张，覆盖视野各区域）
3. 检测角点：`cv2.findChessboardCorners`
4. 亚像素精化：`cv2.cornerSubPix`
5. 求解标定：`cv2.calibrateCamera`

### OpenCV 示例

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,   # 3D 角点坐标列表
    imgpoints,   # 2D 图像角点列表
    img_size,
    None, None
)
# mtx: 内参矩阵, dist: 畸变系数
```

### 去畸变

```python
undistorted = cv2.undistort(img, mtx, dist, None, mtx)
```

## 评估指标

- **重投影误差（RMS）**：< 0.5 pixel 为优，< 1.0 pixel 可用
