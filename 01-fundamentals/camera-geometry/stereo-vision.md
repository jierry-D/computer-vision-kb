# 双目视觉

## 原理

通过两个相机的视差（disparity）反推深度：

```
depth Z = (f × B) / disparity
```

- `f`：焦距（像素）
- `B`：基线（两相机光心距离）
- `disparity`：同一点在左右图像的像素偏移量

## 流程

1. **双目标定**：分别标定左右相机，再联合标定得到 R, T
2. **图像矫正（Rectification）**：使对应点在同一水平线上
3. **视差计算**：SGBM / BM 算法，或深度学习方法
4. **深度图生成**：由视差图换算

## OpenCV 示例

```python
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=9)
disparity = stereo.compute(img_left_gray, img_right_gray)
depth = (focal_length * baseline) / disparity
```

## 深度学习方法

- PSMNet、RAFT-Stereo、CREStereo
