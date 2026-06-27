# 传统特征提取

## 特征描述子对比

| 特征 | 尺度不变 | 旋转不变 | 速度 | 用途 |
|------|---------|---------|------|------|
| SIFT | ✅ | ✅ | 慢 | 图像匹配、拼接 |
| SURF | ✅ | ✅ | 中 | SIFT 加速版 |
| ORB | ✅ | ✅ | 快 | 实时 SLAM、AR |
| BRIEF | ❌ | ❌ | 极快 | 需配合检测器 |
| HOG | ❌ | 部分 | 中 | 行人检测 |
| LBP | ❌ | ✅ | 快 | 纹理、人脸识别 |

## OpenCV 示例（ORB）

```python
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(img, None)

# BF 匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
```

## 注意

- SIFT/SURF 曾有专利限制，OpenCV 4.x 已开放
- 深度学习特征（SuperPoint、DISK）已逐步替代传统特征
