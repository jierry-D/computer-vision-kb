# OpenCV 笔记

## 安装

```bash
pip install opencv-python          # 不含 contrib
pip install opencv-contrib-python  # 含 SIFT/SURF 等扩展
```

## 常用操作速查

```python
import cv2
import numpy as np

# 读写
img = cv2.imread("image.jpg")          # BGR
img = cv2.imread("image.jpg", 0)       # 灰度
cv2.imwrite("out.jpg", img)

# 颜色空间
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 缩放
resized = cv2.resize(img, (640, 480))
resized = cv2.resize(img, None, fx=0.5, fy=0.5)

# 裁剪（numpy 切片）
roi = img[y1:y2, x1:x2]

# 画框/文字
cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
cv2.putText(img, "label", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# 显示
cv2.imshow("window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 视频处理

```python
cap = cv2.VideoCapture("video.mp4")  # 或 0 表示摄像头
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    # process frame...
cap.release()
```
