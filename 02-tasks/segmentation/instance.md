# 实例分割

> 同时检测目标并生成逐实例的像素级掩码，区分同类不同实例。

## 主流方法

### Two-Stage（先检测后分割）

| 模型 | 特点 |
|------|------|
| Mask R-CNN | 实例分割开山之作，ROI Align + 掩码头 |
| PANet | Path Aggregation，特征融合改进 |
| HTC | 级联检测+分割，精度高 |

### One-Stage

| 模型 | 特点 |
|------|------|
| YOLACT | 实时实例分割，线性组合掩码 |
| SOLOv2 | 按位置分类，无需检测框 |
| CondInst | 动态卷积核生成掩码 |

### Query-Based

| 模型 | 特点 |
|------|------|
| QueryInst | 基于 DETR 的实例分割 |
| Mask2Former | 统一框架，query 匹配 |
| SAM | Meta 通用分割大模型，zero-shot |

---

## Mask R-CNN：ROI Align 原理与实现

### ROI Align vs ROI Pooling

**ROI Pooling** 存在两次量化误差：
1. 将 RoI 坐标映射到特征图时取整
2. 将特征图分格时取整

**ROI Align** 消除量化误差，使用双线性插值精确采样：

```
RoI [x1, y1, x2, y2]（原图坐标）
   ↓ 除以步长（如 16），得到特征图坐标（保留小数）
   ↓ 将 RoI 等分为 output_h × output_w 个 bin
   ↓ 每个 bin 内均匀采样 4 个点（位置保留小数）
   ↓ 双线性插值从特征图取值
   ↓ Max/Average Pooling 得到每个 bin 的值
```

```python
import torch
import torch.nn.functional as F

def roi_align_single(feature, roi, output_size, spatial_scale=1/16, sampling_ratio=2):
    """
    feature: [C, H, W]
    roi:     [x1, y1, x2, y2]（原图坐标）
    output_size: (out_h, out_w)
    """
    C, H, W = feature.shape
    out_h, out_w = output_size

    # 映射到特征图坐标
    x1 = roi[0] * spatial_scale
    y1 = roi[1] * spatial_scale
    x2 = roi[2] * spatial_scale
    y2 = roi[3] * spatial_scale

    roi_h = max(y2 - y1, 1.0)
    roi_w = max(x2 - x1, 1.0)

    bin_h = roi_h / out_h
    bin_w = roi_w / out_w

    output = torch.zeros(C, out_h, out_w)
    n_pts = sampling_ratio  # 每个 bin 每维度的采样点数

    for i in range(out_h):
        for j in range(out_w):
            # bin 内采样点坐标
            vals = []
            for pi in range(n_pts):
                for pj in range(n_pts):
                    ys = y1 + (i + (pi + 0.5) / n_pts) * bin_h
                    xs = x1 + (j + (pj + 0.5) / n_pts) * bin_w
                    # 归一化到 [-1, 1]（grid_sample 格式）
                    yn = (ys / (H - 1)) * 2 - 1
                    xn = (xs / (W - 1)) * 2 - 1
                    grid = torch.tensor([[[[xn, yn]]]]).float()
                    # 双线性插值
                    v = F.grid_sample(feature.unsqueeze(0), grid,
                                      mode='bilinear', align_corners=True)
                    vals.append(v.squeeze())
            output[:, i, j] = torch.stack(vals).mean(0)

    return output
```

**踩坑记录**：`torchvision.ops.roi_align` 的 `aligned=True` 参数与论文实现完全一致，建议生产代码直接用官方实现。

---

## SOLOv2：动态卷积核生成过程

SOLOv2 的核心思想：**按位置和尺寸分类**，用动态生成的卷积核直接预测掩码。

```
输入特征图 [B, C, H, W]
   ↓
分类分支（Category Branch）：
  [B, S×S, num_classes]  # S×S 网格，每格预测类别
   ↓
核生成分支（Kernel Branch）：
  [B, S×S, D]            # D=256，每格对应一个卷积核向量
   ↓
掩码特征（Mask Feature）：
  [B, E, H/4, W/4]       # E=128，全图掩码特征图
   ↓
逐实例动态卷积（每个核与掩码特征做点积）：
  kernel[i] ★ mask_feat → mask[i]  # [H/4, W/4]
```

```python
# 动态卷积核应用（矩阵乘法实现批量推理）
def dynamic_conv(mask_feat, kernels):
    """
    mask_feat: [1, E, H, W]
    kernels:   [N, E]，N 个实例的核向量
    返回:      [N, H, W]
    """
    N, E = kernels.shape
    _, _, H, W = mask_feat.shape
    # 展平特征 [E, H*W]
    feat_flat = mask_feat.reshape(E, H * W)
    # 矩阵乘法 [N, E] × [E, H*W] → [N, H*W]
    masks_flat = kernels @ feat_flat
    return masks_flat.reshape(N, H, W).sigmoid()
```

---

## SAM（Segment Anything Model）三种提示模式

SAM 支持三种交互提示方式：

### 1. 点提示

```python
from segment_anything import SamPredictor, sam_model_registry
import numpy as np

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)  # 编码图像特征（只需一次）

# 前景点（label=1）+ 背景点（label=0）
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375], [600, 300]]),
    point_labels=np.array([1, 0]),
    multimask_output=True,   # 返回多个候选 mask
)
best_mask = masks[scores.argmax()]
```

### 2. 框提示

```python
masks, scores, logits = predictor.predict(
    box=np.array([x1, y1, x2, y2]),  # xyxy 格式
    multimask_output=False,
)
```

### 3. 批量使用（SamAutomaticMaskGenerator）

```python
from segment_anything import SamAutomaticMaskGenerator

generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,         # 采样点网格密度
    pred_iou_thresh=0.86,       # IoU 阈值过滤
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,   # 过滤小噪声区域
)
masks = generator.generate(image)
# 每个 mask 包含：segmentation, area, bbox, predicted_iou 等字段
```

---

## COCO 实例分割评估脚本

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# 加载 GT 标注
coco_gt = COCO('annotations/instances_val2017.json')

# 加载模型预测结果（COCO 格式）
# 每条预测：{"image_id": 123, "category_id": 1,
#            "segmentation": RLE, "score": 0.9, "bbox": [x,y,w,h]}
with open('predictions.json') as f:
    preds = json.load(f)
coco_dt = coco_gt.loadRes(preds)

# 运行评估
coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')  # 'segm' 或 'bbox'
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
# 输出：AP, AP50, AP75, APs, APm, APl
```

---

## Mask 后处理：形态学优化

```python
import cv2
import numpy as np

def refine_mask(binary_mask):
    """
    对二值掩码做形态学后处理：
    1. 填充内部孔洞
    2. 去除细小噪声
    3. 平滑边界
    """
    mask = binary_mask.astype(np.uint8) * 255
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # 1. 闭运算：填充小孔（先膨胀后腐蚀）
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel7, iterations=2)
    # 2. 开运算：去除细小噪声（先腐蚀后膨胀）
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    # 3. 高斯模糊 + 重新二值化（平滑锯齿边界）
    mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask_refined = cv2.threshold(mask_blur, 127, 255, cv2.THRESH_BINARY)

    return mask_refined // 255

def keep_largest_component(binary_mask):
    """只保留最大连通区域，去除离散小块"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return binary_mask
    # 标签 0 是背景，跳过
    largest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    return (labels == largest).astype(np.uint8)
```

## 评估指标

- **mask AP**：基于掩码 IoU 的 AP（COCO 标准）
