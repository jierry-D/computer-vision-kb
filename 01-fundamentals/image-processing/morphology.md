# 形态学操作

> 基于结构元素（kernel）对二值/灰度图进行形状分析。

## 基本操作

| 操作 | 效果 | 用途 |
|------|------|------|
| 腐蚀 Erosion | 缩小前景，去小噪点 | 去除小白点 |
| 膨胀 Dilation | 扩大前景，填小孔 | 连接断裂区域 |
| 开运算 Opening | 腐蚀→膨胀，去小目标 | 消除小噪声 |
| 闭运算 Closing | 膨胀→腐蚀，填小孔洞 | 填充内部空洞 |
| 梯度 Gradient | 膨胀-腐蚀 | 提取边缘 |
| 顶帽 Top Hat | 原图-开运算 | 提取亮细节 |
| 黑帽 Black Hat | 闭运算-原图 | 提取暗细节 |

## OpenCV 示例

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded = cv2.erode(img, kernel, iterations=1)
dilated = cv2.dilate(img, kernel, iterations=1)
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```
