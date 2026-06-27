# OpenMMLab 框架

> 商汤+旷视开源的视觉算法框架体系，覆盖几乎所有视觉任务。

## 主要框架

| 框架 | 任务 |
|------|------|
| MMDetection | 目标检测、实例分割 |
| MMSegmentation | 语义分割 |
| MMPose | 姿态估计 |
| MMTracking | 目标跟踪 |
| MMClassification（mmpretrain）| 图像分类、自监督 |
| MMDeploy | 模型部署（TRT/ONNX/ncnn） |
| MMRotate | 旋转目标检测 |

## 安装（以 MMDetection 为例）

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

## 快速推理

```python
from mmdet.apis import init_detector, inference_detector
model = init_detector("config.py", "checkpoint.pth", device="cuda:0")
result = inference_detector(model, "image.jpg")
```

## 训练

```bash
python tools/train.py configs/yolox/yolox_s_8x8_300e_coco.py
```

## 提示

- 配置文件继承机制：`_base_` 字段，改动最小化
- 所有框架统一切换到 MMEngine 作为训练引擎（v2.x 之后）

---

## MMDetection 配置文件详解

MMDetection 的配置文件是一个 Python 字典，结构清晰但初看较复杂。以 Faster R-CNN 为例：

```python
# configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py（简化版）

# === 模型结构 ===
model = dict(
    type='FasterRCNN',

    # 骨干网络（特征提取）
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),   # 输出哪几个 stage 的特征
        frozen_stages=1,             # 冻结前几个 stage（减少计算，防止过拟合）
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),

    # 特征金字塔网络（多尺度特征融合）
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # backbone 各 stage 输出通道
        out_channels=256,
        num_outs=5,
    ),

    # 区域提议网络（Region Proposal Network）
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder', ...),
        loss_cls=dict(type='CrossEntropyLoss', ...),
        loss_bbox=dict(type='L1Loss', ...),
    ),

    # 检测头（分类+回归）
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,           # COCO 类别数
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        ),
    ),

    # 训练配置
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5),
        ),
        rcnn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5),
            sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25),
        ),
    ),
    # 测试配置
    test_cfg=dict(
        rpn=dict(nms=dict(type='nms', iou_threshold=0.7), max_per_img=1000),
        rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100),
    ),
)

# === 数据集 ===
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# === 训练超参 ===
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', milestones=[8, 11], gamma=0.1, by_epoch=True),
]
```

---

## 自定义数据集接入步骤

### 方法一：继承 BaseDetDataset（推荐，MMDet v3+）

```python
# my_dataset.py
import os
from mmdet.registry import DATASETS
from mmdet.datasets import BaseDetDataset

@DATASETS.register_module()
class MyCustomDataset(BaseDetDataset):
    """自定义数据集，COCO JSON 格式"""

    METAINFO = {
        'classes': ('crack', 'rust', 'deformation'),  # 类别名称
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142)]  # 可视化颜色
    }

    def load_data_list(self):
        """加载数据列表，返回标准格式的 dict 列表"""
        import json
        with open(self.ann_file) as f:
            coco_data = json.load(f)

        # 构建 image_id → annotations 映射
        img_id2anns = {}
        for ann in coco_data['annotations']:
            img_id2anns.setdefault(ann['image_id'], []).append(ann)

        data_list = []
        for img_info in coco_data['images']:
            img_id = img_info['id']
            data_info = {
                'img_path': os.path.join(self.data_prefix['img'], img_info['file_name']),
                'img_id': img_id,
                'height': img_info['height'],
                'width': img_info['width'],
                'instances': []
            }
            for ann in img_id2anns.get(img_id, []):
                x, y, w, h = ann['bbox']
                data_info['instances'].append({
                    'bbox': [x, y, x+w, y+h],     # 转为 xyxy 格式
                    'bbox_label': ann['category_id'] - 1,  # 0-indexed
                    'ignore_flag': ann.get('iscrowd', 0)
                })
            data_list.append(data_info)

        return data_list
```

```python
# 在配置文件中使用自定义数据集
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='MyCustomDataset',
        data_root='data/my_dataset/',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        pipeline=[...],
    )
)
```

---

## 多卡训练命令

```bash
# 单机多卡（推荐，使用 dist_train.sh）
bash tools/dist_train.sh \
    configs/yolox/yolox_s_8x8_300e_coco.py \
    8 \                                        # GPU 数量
    --work-dir work_dirs/yolox_s_coco

# 等价的 torchrun 命令（更现代）
torchrun --nproc_per_node=8 \
    tools/train.py \
    configs/yolox/yolox_s_8x8_300e_coco.py \
    --launcher pytorch \
    --work-dir work_dirs/yolox_s_coco

# 继续训练（从 checkpoint 恢复）
bash tools/dist_train.sh config.py 8 \
    --resume work_dirs/yolox_s_coco/epoch_10.pth

# 仅评估
bash tools/dist_test.sh config.py checkpoint.pth 8
```

⚠️ **注意**：多卡训练时 batch size 是每卡的 batch size。8 卡 × 8 per GPU = 总 batch size 64。学习率通常需要线性缩放：`lr = base_lr * total_batch_size / 8`。

---

## 模型调试技巧（打印中间特征图）

```python
# 方法一：使用 PyTorch hook 打印特征图统计信息
from mmdet.apis import init_detector
import torch

model = init_detector("config.py", "checkpoint.pth", device="cuda:0")

feature_maps = {}

def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            feature_maps[name] = output.detach()
            print(f"{name}: shape={output.shape}, "
                  f"mean={output.mean():.4f}, std={output.std():.4f}, "
                  f"min={output.min():.4f}, max={output.max():.4f}")
    return hook

# 注册 hook
for name, module in model.named_modules():
    if 'backbone' in name and isinstance(module, torch.nn.Conv2d):
        module.register_forward_hook(hook_fn(name))

# 前向传播触发 hook
result = model.test_step({"img": ..., "data_samples": ...})
```

```python
# 方法二：可视化特征图（适合快速观察）
import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_map(feat, n_channels=16, figsize=(20, 10)):
    """可视化特征图的前 n_channels 个通道"""
    feat = feat[0].cpu().numpy()  # 取 batch 第一张
    n = min(n_channels, feat.shape[0])
    fig, axes = plt.subplots(2, n // 2, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(feat[i], cmap='viridis')
            ax.set_title(f"Channel {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 使用
visualize_feature_map(feature_maps['backbone.layer2.1.conv2'])
```

---

## MMDeploy 转 TRT 完整命令流程

```bash
# 1. 安装 MMDeploy
pip install mmdeploy
pip install mmdeploy-runtime-gpu  # GPU 推理运行时

# 2. 导出模型（PyTorch → TensorRT）
python tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \  # 部署配置
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \                       # 模型配置
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \                              # 权重
    tests/data/tiger.jpeg \                                                     # 测试图
    --work-dir work_dirs/faster_rcnn_trt \                                      # 输出目录
    --device cuda:0 \
    --dump-info                                                                 # 保存模型信息

# 3. 生成文件说明
# work_dirs/faster_rcnn_trt/
# ├── end2end.engine   # TensorRT engine 文件
# ├── deploy.json      # 部署配置
# └── pipeline.json    # 推理流程

# 4. TensorRT 速度测试
python tools/profiler.py \
    work_dirs/faster_rcnn_trt/deploy.json \
    work_dirs/faster_rcnn_trt/pipeline.json \
    tests/data/ \
    --device cuda:0 \
    --num-iter 100
```

```python
# 5. Python 推理
from mmdeploy_runtime import Detector

detector = Detector(
    model_path="work_dirs/faster_rcnn_trt",
    device_name="cuda",
    device_id=0
)

import cv2
img = cv2.imread("test.jpg")
bboxes, labels, masks = detector(img)
print(f"检测到 {len(bboxes)} 个目标")
for bbox, label in zip(bboxes, labels):
    x1, y1, x2, y2, score = bbox
    print(f"类别 {label}: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}), 置信度={score:.3f}")
```
