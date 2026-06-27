# 边缘检测

## 方法对比

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| Sobel | 一阶差分 | 快速，抗噪 | 边缘粗 |
| Scharr | 改进 Sobel | 精度更高 | — |
| Laplacian | 二阶差分 | 各向同性 | 对噪声敏感 |
| Canny | 多步骤最优 | 细边缘，双阈值 | 参数较多 |

## Canny 流程

1. 高斯平滑（去噪）
2. 计算梯度幅值与方向
3. 非极大值抑制（NMS）
4. 双阈值筛选（高阈值确定强边缘，低阈值延伸弱边缘）
5. 边缘连接

## OpenCV 示例

```python
edges = cv2.Canny(img, threshold1=50, threshold2=150)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
```
