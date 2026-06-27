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

---

## 图像几何变换

### 仿射变换（Affine Transform）

保持直线和平行性，支持平移、旋转、缩放、倾斜。

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")
h, w = img.shape[:2]

# 1. 旋转（绕中心旋转 45 度，缩放 0.8）
center = (w // 2, h // 2)
M_rot = cv2.getRotationMatrix2D(center, angle=45, scale=0.8)
rotated = cv2.warpAffine(img, M_rot, (w, h))

# 2. 三点仿射变换（从源三点到目标三点的映射）
src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1]])
dst_pts = np.float32([[50, 30], [w-30, 10], [20, h-50]])
M_aff = cv2.getAffineTransform(src_pts, dst_pts)
affine = cv2.warpAffine(img, M_aff, (w, h))
```

### 透视变换（Perspective Transform）

将四边形区域变换为矩形，常用于文档矫正、车牌识别。

```python
# 四点透视变换（将梯形区域变为矩形）
src_pts = np.float32([
    [100, 50],   # 左上
    [400, 50],   # 右上
    [450, 350],  # 右下
    [50, 350],   # 左下
])
dst_pts = np.float32([
    [0, 0],
    [300, 0],
    [300, 300],
    [0, 300],
])
M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M_persp, (300, 300))

# 逆变换（将矩形区域映射回原始视角）
M_inv = np.linalg.inv(M_persp)
restored = cv2.warpPerspective(warped, M_inv, (w, h))
```

---

## 直方图均衡化与 CLAHE

```python
img = cv2.imread("low_contrast.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. 全局直方图均衡化（整体对比度增强，可能过度增强）
equalized = cv2.equalizeHist(gray)

# 2. CLAHE（对比度受限的自适应直方图均衡化，推荐）
clahe = cv2.createCLAHE(
    clipLimit=2.0,       # 对比度上限，防止噪声放大
    tileGridSize=(8, 8)  # 图像分块大小
)
clahe_result = clahe.apply(gray)

# 3. 彩色图像的 CLAHE（在 LAB 空间只处理亮度通道）
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l_clahe = clahe.apply(l)
lab_clahe = cv2.merge([l_clahe, a, b])
result_color = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# 可视化对比
comparison = np.hstack([gray, equalized, clahe_result])
cv2.imshow("原始 | 全局均衡 | CLAHE", comparison)
cv2.waitKey(0)
```

💡 **Tips**：CLAHE 比全局直方图均衡化效果更好，尤其适合工业检测中光照不均的场景。`clipLimit` 越小，限制越严，防噪效果越好但增强效果越弱，通常取 1.5-3.0。

---

## 轮廓检测与形状分析

```python
img = cv2.imread("shapes.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# 或自适应阈值
binary_adap = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, blockSize=11, C=2
)

# 形态学操作（去噪）
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 填充小洞
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # 去除小噪点

# 查找轮廓
contours, hierarchy = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
print(f"检测到 {len(contours)} 个轮廓")

# 分析每个轮廓
result = img.copy()
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)          # 面积（像素）
    perimeter = cv2.arcLength(cnt, True) # 周长（像素）
    if area < 100:                        # 过滤小轮廓
        continue

    # 外接矩形（axis-aligned）
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 最小外接矩形（旋转矩形）
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(result, [box], 0, (0, 0, 255), 2)

    # 外接圆
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    cv2.circle(result, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)

    # 圆形度（越接近 1 越圆）
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)

    print(f"轮廓 {i}: 面积={area:.0f} 周长={perimeter:.1f} 圆形度={circularity:.3f}")

cv2.imshow("轮廓检测", result)
cv2.waitKey(0)
```

---

## 模板匹配

```python
import cv2
import numpy as np

# 加载源图和模板
scene = cv2.imread("scene.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)
h, w = template.shape

# 模板匹配（多种方法）
method = cv2.TM_CCOEFF_NORMED  # 推荐，归一化相关系数，结果在 [-1, 1]
result = cv2.matchTemplate(scene, template, method)

# 找最佳匹配位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc  # TM_CCOEFF_NORMED 取最大值
bottom_right = (top_left[0] + w, top_left[1] + h)
print(f"最佳匹配置信度: {max_val:.3f}, 位置: {top_left}")

# 画出匹配结果
scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
cv2.rectangle(scene_color, top_left, bottom_right, (0, 255, 0), 2)

# 多目标匹配（找到所有超过阈值的位置）
threshold = 0.8
locations = np.where(result >= threshold)
for pt in zip(*locations[::-1]):  # 注意 x,y 顺序
    cv2.rectangle(scene_color, pt, (pt[0]+w, pt[1]+h), (0, 0, 255), 1)
```

💡 **Tips**：模板匹配对旋转和缩放变化敏感。如果目标有旋转，可以用多角度模板或特征点匹配（SIFT/ORB）代替。

---

## 视频写出（VideoWriter）

```python
import cv2

# 输入视频
cap = cv2.VideoCapture("input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义输出视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # MP4 格式
# 其他编码：'XVID'（AVI）, 'MJPG'（MJPEG）, 'H264'（需额外编解码器）
out = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 处理帧
    processed = cv2.GaussianBlur(frame, (5, 5), 0)  # 示例：高斯模糊

    # 写入帧
    out.write(processed)

cap.release()
out.release()
print("视频写出完成")
```

⚠️ **注意**：`mp4v` 编码在某些播放器可能无法播放，更好的选择是在安装了 FFmpeg 的系统上使用 `avc1`（H.264）。如果帧不是 BGR 格式，写出前需要转换回 BGR。

---

## 与 PyTorch Tensor 互转

OpenCV 和 PyTorch 之间的格式转换是最常见的坑之一：

```python
import cv2
import torch
import numpy as np

# OpenCV → PyTorch Tensor
def cv2_to_tensor(img_bgr):
    """
    OpenCV BGR uint8 → PyTorch float32 Tensor
    [H, W, C] BGR → [C, H, W] RGB，归一化到 [0, 1]
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR → RGB
    img_float = img_rgb.astype(np.float32) / 255.0      # uint8 → float32
    tensor = torch.from_numpy(img_float).permute(2, 0, 1)  # HWC → CHW
    return tensor

# PyTorch Tensor → OpenCV
def tensor_to_cv2(tensor):
    """
    PyTorch float32 Tensor → OpenCV BGR uint8
    [C, H, W] RGB → [H, W, C] BGR
    """
    img = tensor.detach().cpu().numpy()           # 脱离计算图，移到 CPU
    img = np.transpose(img, (1, 2, 0))            # CHW → HWC
    img = (img * 255).clip(0, 255).astype(np.uint8)  # float → uint8
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # RGB → BGR
    return img_bgr

# 示例
img = cv2.imread("image.jpg")
tensor = cv2_to_tensor(img)
print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
# torch.Size([3, 480, 640]), torch.float32

# 加 batch 维度（模型推理）
batch = tensor.unsqueeze(0)  # [1, C, H, W]

# 推理后转回
with torch.no_grad():
    output = model(batch.cuda())
result_img = tensor_to_cv2(output.squeeze(0).cpu())
```

---

## 常见坑总结

| 坑 | 说明 | 解决 |
|----|------|------|
| BGR vs RGB | OpenCV 读取为 BGR，PyTorch/PIL 使用 RGB | 转换时使用 `cv2.COLOR_BGR2RGB` |
| 数据类型不匹配 | OpenCV 操作多数要求 `uint8`，深度学习需要 `float32` | 用 `.astype()` 明确转换 |
| 通道顺序（HWC vs CHW） | OpenCV 为 HWC，PyTorch 为 CHW | 用 `transpose()` 或 `.permute()` |
| 显示时自动缩放 | `imshow` 会自动缩放 float 图（0-1 映射为白色）| 显示前转为 uint8 |
| `imread` 返回 None | 路径错误或文件损坏 | 加空值检查 `if img is None: raise` |
| 视频写出全黑 | 输入帧尺寸与 VideoWriter 初始化尺寸不一致 | 统一 resize 后再写 |
| SIFT 不可用 | `opencv-python` 不含专利算法 | 换用 `opencv-contrib-python` |
