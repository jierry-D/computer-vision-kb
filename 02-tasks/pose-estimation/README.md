# 姿态估计

## 任务类型

| 类型 | 输出 | 典型应用 |
|------|------|---------|
| 人体关键点检测（2D） | 17/133 个关节点坐标 | 动作识别、健身 |
| 人体关键点检测（3D） | 3D 关节点坐标 | 虚拟试衣、动画 |
| 手部关键点检测 | 21 个关节点 | AR/手势交互 |
| 6DoF 物体姿态 | 3D 平移 + 旋转 | 机器人抓取 |

## 2D 人体姿态主流模型

### Top-Down（先检测人，再估计关键点）

| 模型 | 特点 |
|------|------|
| HRNet | 高分辨率网络，精度高 |
| SimpleBaseline | 简洁高效，强基准 |
| ViTPose | ViT 骨干，SOTA |

### Bottom-Up（先检测所有关键点，再分组）

| 模型 | 特点 |
|------|------|
| OpenPose | 经典，PAF 关联场 |
| HigherHRNet | 多尺度 HRNet |

## 6DoF 姿态估计

| 模型 | 方法 |
|------|------|
| PVNet | 关键点投票 |
| GDR-Net | 几何引导回归 |
| FoundPose | 基础模型 zero-shot |

## 推荐框架

- **MMPose**：OpenMMLab，覆盖所有姿态任务

---

## Heatmap-based 关键点检测原理

主流 2D 姿态估计通过预测**高斯热图**来定位关键点，而非直接回归坐标。

### 高斯热图生成

```python
import numpy as np
import cv2

def generate_heatmap(h, w, joint_x, joint_y, sigma=2):
    """
    生成单个关键点的高斯热图
    h, w:      热图尺寸（通常是原图 1/4）
    joint_x/y: 关键点在热图上的坐标
    sigma:     高斯核标准差（控制范围大小）
    """
    heatmap = np.zeros((h, w), dtype=np.float32)
    if joint_x < 0 or joint_y < 0:  # 关键点不可见
        return heatmap

    # 生成高斯核
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0, y0 = size // 2, size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # 在关键点周围绘制高斯核（处理边界裁剪）
    ul = [int(joint_x - size // 2), int(joint_y - size // 2)]
    br = [int(joint_x + size // 2 + 1), int(joint_y + size // 2 + 1)]

    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)

    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return heatmap

def generate_all_heatmaps(image_size, keypoints, num_joints=17, sigma=2, stride=4):
    """
    生成所有关节点的热图堆叠
    keypoints: [num_joints, 3]，每行 (x, y, visibility)
    返回:      [num_joints, H/stride, W/stride]
    """
    h, w = image_size[0] // stride, image_size[1] // stride
    heatmaps = np.zeros((num_joints, h, w), dtype=np.float32)
    for i, (x, y, v) in enumerate(keypoints):
        if v > 0:  # 可见关键点
            heatmaps[i] = generate_heatmap(h, w, x / stride, y / stride, sigma)
    return heatmaps
```

### 热图解码（argmax + 亚像素偏移）

```python
def decode_heatmaps(heatmaps, stride=4):
    """
    heatmaps: [num_joints, H, W]
    返回:     [num_joints, 3]，(x, y, score) 原图坐标
    """
    num_joints, H, W = heatmaps.shape
    keypoints = np.zeros((num_joints, 3))

    for i in range(num_joints):
        hm = heatmaps[i]
        # 找最大值位置
        flat_idx = hm.argmax()
        y, x = divmod(flat_idx, W)
        score = hm[y, x]

        # 亚像素精度：向梯度方向微移 0.25 像素
        if 1 <= x < W - 1 and 1 <= y < H - 1:
            dx = np.sign(hm[y, x + 1] - hm[y, x - 1])
            dy = np.sign(hm[y + 1, x] - hm[y - 1, x])
            x += dx * 0.25
            y += dy * 0.25

        keypoints[i] = [x * stride, y * stride, score]
    return keypoints
```

---

## HRNet：并行多分辨率维护机制

传统网络（ResNet/VGG）逐步下采样，高分辨率信息丢失。HRNet 通过**并行维护多个分辨率分支**并反复融合，始终保持高分辨率表示：

```
输入 [B, 3, 256, 192]
   ↓
Stage 1: 分支1（H/4, 64c）
   ↓
Stage 2: 分支1（H/4, 48c）
         分支2（H/8, 96c）
         ← 双向融合 →
   ↓
Stage 3: 分支1（H/4, 48c）
         分支2（H/8, 96c）
         分支3（H/16, 192c）
         ← 三向融合 →
   ↓
Stage 4: 四分支并行 + 融合
   ↓
取最高分辨率分支 → 关键点热图预测头
输出: [B, num_joints, H/4, W/4]
```

**分支融合策略**：
- 低 → 高分辨率：双线性上采样 + 1×1 卷积
- 高 → 低分辨率：步长为 2 的 3×3 卷积

---

## ViTPose：ViT 骨干适配方案

ViTPose 直接用 Vision Transformer 作为姿态估计骨干，通过以下方式适配：

1. **输入处理**：将图像划分为 16×16 的 patch（patch embed）
2. **特征提取**：ViT-B/L/H 各层 self-attention 输出
3. **解码器**：轻量级解码器（简单反卷积头），将 patch 特征上采样到 H/4 大小
4. **预训练**：使用 MAE 预训练骨干，再在 COCO 上微调

```python
# 使用 MMPose 加载 ViTPose
from mmpose.apis import init_model, inference_topdown

config = 'configs/body_2d_keypoint/topdown_heatmap/coco/vitpose-b_8xb64-210e_coco-256x192.py'
checkpoint = 'vitpose-b_simcc-coco_pt-aic-coco_420e-256x192.pth'
model = init_model(config, checkpoint, device='cuda:0')

results = inference_topdown(model, 'person.jpg',
                             bboxes=[[100, 50, 400, 600]])  # xyxy
for result in results:
    keypoints = result.pred_instances.keypoints    # [1, 17, 2]
    scores    = result.pred_instances.keypoint_scores  # [1, 17]
```

---

## 人体姿态骨架可视化（OpenCV）

```python
import cv2
import numpy as np

# COCO 17 关键点名称
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO 骨骼连接定义（pair of keypoint indices）
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),         # 头部
    (5, 6),                                   # 肩膀
    (5, 7), (7, 9), (6, 8), (8, 10),         # 手臂
    (5, 11), (6, 12), (11, 12),              # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16), # 腿部
]

# 各骨骼线段颜色（BGR）
SKELETON_COLORS = [
    (255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
    (255, 153, 255), (153, 204, 255), (255, 102, 255), (255, 51, 255),
    (102, 178, 255), (51, 153, 255), (255, 153, 153), (255, 102, 102),
    (255, 51, 51), (153, 255, 153), (102, 255, 102), (51, 255, 51),
]

def draw_skeleton(image, keypoints, scores, score_threshold=0.3):
    """
    image:     [H, W, 3]，BGR 图像
    keypoints: [17, 2]，(x, y) 像素坐标
    scores:    [17]，置信度
    """
    img = image.copy()

    # 画骨骼连线
    for idx, (i, j) in enumerate(COCO_SKELETON):
        if scores[i] > score_threshold and scores[j] > score_threshold:
            pt1 = tuple(keypoints[i].astype(int))
            pt2 = tuple(keypoints[j].astype(int))
            color = SKELETON_COLORS[idx % len(SKELETON_COLORS)]
            cv2.line(img, pt1, pt2, color, thickness=2)

    # 画关键点
    for i, (x, y) in enumerate(keypoints):
        if scores[i] > score_threshold:
            cv2.circle(img, (int(x), int(y)), radius=4,
                       color=(0, 255, 0), thickness=-1)

    return img
```

---

## 6DoF 姿态：PnP 求解代码

已知 3D 模型关键点坐标和图像中对应的 2D 投影坐标，用 PnP 求解 6DoF 姿态（R, T）：

```python
import cv2
import numpy as np

def solve_pose_pnp(points_3d, points_2d, camera_matrix, dist_coeffs=None):
    """
    points_3d:     [N, 3]，模型坐标系下的 3D 点（float32）
    points_2d:     [N, 2]，图像坐标系下的 2D 投影点（float32）
    camera_matrix: [3, 3]，相机内参矩阵 K
    dist_coeffs:   [4/5]，畸变系数（无畸变则传 None）
    返回:          (rvec, tvec) 旋转向量和平移向量
    """
    points_3d = points_3d.astype(np.float32)
    points_2d = points_2d.astype(np.float32)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))

    # RANSAC PnP（抗离群点）
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, points_2d, camera_matrix, dist_coeffs,
        iterationsCount=100, reprojectionError=8.0,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        return None, None

    # 精化（用 Levenberg-Marquardt 迭代优化）
    rvec, tvec = cv2.solvePnPRefineLM(
        points_3d[inliers.flatten()],
        points_2d[inliers.flatten()],
        camera_matrix, dist_coeffs, rvec, tvec
    )
    return rvec, tvec

def project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs=None):
    """将 3D 点重投影到图像平面，用于验证 PnP 精度"""
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))
    projected, _ = cv2.projectPoints(
        points_3d.astype(np.float32), rvec, tvec, camera_matrix, dist_coeffs
    )
    return projected.squeeze()

# 使用示例
camera_matrix = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float32)

rvec, tvec = solve_pose_pnp(obj_3d_pts, img_2d_pts, camera_matrix)
if rvec is not None:
    R, _ = cv2.Rodrigues(rvec)  # 旋转向量 → 旋转矩阵
    print("旋转矩阵 R:\n", R)
    print("平移向量 T:", tvec.flatten())
    # 重投影误差检验
    reproj = project_points(obj_3d_pts, rvec, tvec, camera_matrix)
    error = np.linalg.norm(reproj - img_2d_pts, axis=1).mean()
    print(f"平均重投影误差: {error:.2f} px")
```

**踩坑记录**：
- PnP 至少需要 4 个非共面的对应点，共面情况下用 `cv2.SOLVEPNP_IPPE` 更稳定
- RANSAC 版本对离群点鲁棒，但 `iterationsCount` 不够多时可能结果不稳定，建议设 300 以上
- 相机内参不准是最常见的 PnP 失败原因，务必先用棋盘格标定
