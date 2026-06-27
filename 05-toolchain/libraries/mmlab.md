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
