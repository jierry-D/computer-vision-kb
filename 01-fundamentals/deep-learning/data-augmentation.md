# 数据增强

## 基础增强

| 方法 | 效果 |
|------|------|
| 随机翻转 | 方向不变性 |
| 随机裁剪/缩放 | 尺度、位置不变性 |
| 颜色抖动（亮度/对比度/饱和度）| 光照不变性 |
| 随机旋转 | 方向泛化 |
| 高斯噪声/模糊 | 鲁棒性 |

## 高级增强

| 方法 | 说明 | 适用 |
|------|------|------|
| Mixup | 两张图线性叠加 | 分类 |
| CutMix | 裁剪区域替换 | 分类/检测 |
| Mosaic | 4 张图拼合 | YOLO 检测 |
| Copy-Paste | 实例粘贴到新背景 | 实例分割 |
| AutoAugment / RandAugment | NAS 搜索最优策略 | 分类 |
| AugMix | 多路增强混合 | 鲁棒性 |

## 推荐库

- **Albumentations**：速度快，支持检测/分割的 bbox/mask 同步增强
- **torchvision.transforms v2**：PyTorch 官方，支持 bbox 同步

```python
import albumentations as A
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Mosaic(p=0.5),
], bbox_params=A.BboxParams(format='yolo'))
```

---

## 完整 Albumentations 配置示例（检测任务含 bbox 同步）

检测任务中增强时必须同步变换 bbox，Albumentations 通过 `BboxParams` 自动处理。

```python
import albumentations as A
import cv2
import numpy as np

# 目标检测任务的完整增强流程
def get_detection_transform(img_size=640, is_train=True):
    if is_train:
        return A.Compose([
            # 几何变换（自动同步 bbox）
            A.RandomResizedCrop(height=img_size, width=img_size,
                                scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                               rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),

            # 颜色/纹理变换（不影响 bbox）
            A.ColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.4, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.ToGray(p=0.05),

            # 遮挡增强
            A.CoarseDropout(max_holes=8, max_height=64, max_width=64,
                            fill_value=0, p=0.3),

            # 归一化
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ],
        bbox_params=A.BboxParams(
            format='yolo',          # 支持 'pascal_voc'/'coco'/'albumentations'/'yolo'
            min_area=256,           # bbox 最小面积（像素），小于此则丢弃
            min_visibility=0.3,     # bbox 可见度阈值，小于此则丢弃
            label_fields=['labels'] # 同步变换 label 字段
        ))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))


# 使用示例
transform = get_detection_transform()
image = cv2.imread('img.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bboxes = [[0.5, 0.5, 0.3, 0.4], [0.2, 0.3, 0.1, 0.15]]  # YOLO格式: [cx,cy,w,h]
labels = [0, 1]

result = transform(image=image, bboxes=bboxes, labels=labels)
aug_image = result['image']
aug_bboxes = result['bboxes']
aug_labels = result['labels']
```

---

## Mosaic 增强实现原理

Mosaic 是 YOLOv4/v5 的核心增强策略，将 **4 张训练图像拼合**成一张，同时增加小目标样本密度。

**步骤**：
1. 随机选取 3 张图像（加上当前图共 4 张）
2. 在拼合图像中随机选取一个分割点 (cx, cy)
3. 将 4 张图分别缩放、裁剪后放入 4 个象限
4. 同步更新每张图的 bbox 坐标（平移 + 裁剪边界检查）

```python
import random
import numpy as np
import cv2

def mosaic_augment(images, bboxes_list, labels_list, output_size=640):
    """
    images: list of 4 numpy arrays (H, W, C)
    bboxes_list: list of 4 arrays, each (N, 4) in [x1,y1,x2,y2] format
    """
    s = output_size
    # 随机选取中心点（偏移在 [s*0.25, s*0.75] 范围内）
    cx = int(random.uniform(s * 0.25, s * 0.75))
    cy = int(random.uniform(s * 0.25, s * 0.75))

    result_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
    result_bboxes, result_labels = [], []

    # 每个子图的放置位置和裁剪逻辑
    placements = [
        (0, 0, cx, cy),         # 左上：子图的右下角放到 (cx,cy)
        (cx, 0, s*2, cy),       # 右上：子图的左下角放到 (cx,cy)
        (0, cy, cx, s*2),       # 左下：子图的右上角放到 (cx,cy)
        (cx, cy, s*2, s*2),     # 右下：子图的左上角放到 (cx,cy)
    ]

    for i, (img, bboxes, labels) in enumerate(zip(images, bboxes_list, labels_list)):
        x1, y1, x2, y2 = placements[i]
        h, w = y2 - y1, x2 - x1
        # 缩放子图到对应区域
        resized = cv2.resize(img, (w, h))
        result_img[y1:y2, x1:x2] = resized

        # 同步变换 bbox（归一化坐标 → 绝对坐标 → 平移）
        if len(bboxes) > 0:
            bx = bboxes.copy().astype(float)
            bx[:, [0, 2]] = bx[:, [0, 2]] * w + x1
            bx[:, [1, 3]] = bx[:, [1, 3]] * h + y1
            result_bboxes.append(bx)
            result_labels.extend(labels)

    # 从 2s×2s 裁剪回 s×s
    result_img = result_img[cy-s//2:cy+s//2, cx-s//2:cx+s//2]
    return result_img, result_bboxes, result_labels
```

---

## Copy-Paste 增强代码示例

Copy-Paste 将一张图的目标实例（带 mask）粘贴到另一张图的随机位置，显著增加实例数量，对实例分割尤其有效（如 COCO 上 +2 mAP）。

```python
import numpy as np
import cv2
import random

def copy_paste(image_dst, masks_dst, labels_dst,
               image_src, masks_src, labels_src,
               paste_prob=0.5):
    """
    将 src 图像中的实例粘贴到 dst 图像
    masks: list of binary masks (H, W)
    """
    h, w = image_dst.shape[:2]
    result = image_dst.copy()
    result_masks = list(masks_dst)
    result_labels = list(labels_dst)

    for mask, label in zip(masks_src, labels_src):
        if random.random() > paste_prob:
            continue
        # 随机缩放和翻转
        scale = random.uniform(0.5, 1.5)
        new_h, new_w = int(mask.shape[0] * scale), int(mask.shape[1] * scale)
        src_region = cv2.resize(image_src, (new_w, new_h))
        src_mask   = cv2.resize(mask.astype(np.uint8), (new_w, new_h)).astype(bool)

        # 随机放置位置
        x_offset = random.randint(0, max(0, w - new_w))
        y_offset = random.randint(0, max(0, h - new_h))
        x_end = min(x_offset + new_w, w)
        y_end = min(y_offset + new_h, h)

        # 抠出 ROI
        roi_mask = src_mask[:y_end-y_offset, :x_end-x_offset]
        # 粘贴（仅在 mask 区域替换像素）
        result[y_offset:y_end, x_offset:x_end][roi_mask] = \
            src_region[:y_end-y_offset, :x_end-x_offset][roi_mask]

        # 更新 mask
        dst_mask = np.zeros((h, w), dtype=bool)
        dst_mask[y_offset:y_end, x_offset:x_end] = roi_mask
        result_masks.append(dst_mask)
        result_labels.append(label)

    return result, result_masks, result_labels
```

---

## 不同任务增强策略建议

| 任务 | 强推增强 | 慎用增强 | 注意事项 |
|------|---------|----------|----------|
| 图像分类 | Mixup、CutMix、RandAugment、ColorJitter | 强几何变换（破坏语义）| label 平滑配合 Mixup |
| 目标检测 | Mosaic、随机裁剪缩放、水平翻转、HSV 调整 | 垂直翻转（部分场景不自然）| bbox 必须同步变换 |
| 语义分割 | 随机裁剪、水平翻转、颜色抖动 | Mixup（标签难以混合）| mask 必须用最近邻插值 |
| 实例分割 | Copy-Paste、随机缩放、水平翻转 | — | mask 精度要求高，旋转角度不宜过大 |
| 关键点检测 | 水平翻转（需重定义左右关键点）、小幅旋转 | 强透视变换 | 关键点坐标必须同步 |

---

## 医学图像增强特殊注意事项

医学图像（CT/MRI/病理切片）有其特殊性，不能直接套用自然图像的增强策略：

1. **颜色空间**：
   - CT 图像为单通道灰度值（HU 单位），勿使用 RGB 颜色抖动
   - 病理切片（H&E 染色）可以用颜色增强，但需保持染色合理性（HED 颜色空间增强更合适）

2. **翻转方向**：
   - 部分解剖结构有方向性（如心脏位置），垂直翻转需谨慎
   - 腹部 CT 不建议上下翻转

3. **强度归一化**：
   - CT 需先窗宽/窗位（windowing）处理，再做 minmax 归一化
   - 增强时不要改变 HU 值的绝对范围

4. **推荐增强**：
   - 弹性形变（Elastic Deformation）：模拟软组织形变，常用于分割
   - 随机 Gamma 调整：调整对比度（适用于 MRI）
   - 随机高斯噪声/模糊：模拟扫描噪声

```python
# 医学图像推荐的 Albumentations 配置
medical_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ElasticTransform(alpha=120, sigma=120*0.05,
                       alpha_affine=120*0.03, p=0.5),
    A.GridDistortion(p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.GaussianBlur(p=0.2),
    A.GaussNoise(p=0.2),
    # 不使用 ColorJitter（CT 灰度图）
])
```

5. **数据量稀缺的应对方案**：
   - 使用迁移学习（ImageNet 预训练权重，将单通道复制为 3 通道）
   - 考虑 Diffusion 模型或 GAN 合成数据
   - k-fold 交叉验证，充分利用有限标注
