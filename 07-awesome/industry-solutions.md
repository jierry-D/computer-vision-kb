# 工业落地方案

## 工业质检

| 项目 | 特点 |
|------|------|
| [anomalib](https://github.com/openvinotoolkit/anomalib) | 异常检测完整流程 |
| [YOLO-World](https://github.com/AILab-CVC/YOLO-World) | 开放词汇检测，免标注 |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | 文本引导检测 |

## OCR

| 项目 | 特点 |
|------|------|
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 中文最强，工业首选 |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) | 80+ 语言，开箱即用 |
| [surya](https://github.com/VikParuchuri/surya) | 文档 OCR，多语言 |

## 人脸

| 项目 | 特点 |
|------|------|
| [InsightFace](https://github.com/deepinsight/insightface) | 人脸识别工业标准 |
| [FaceBoxes / RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) | 人脸检测 |

## 自动驾驶感知

| 项目 | 特点 |
|------|------|
| [BEVFusion](https://github.com/mit-han-lab/bevfusion) | 多模态 BEV 感知 |
| [UniAD](https://github.com/OpenDriveLab/UniAD) | 端到端自动驾驶 |
| [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) | 3D 车道线 |

## 通用视觉基础模型

| 项目 | 特点 |
|------|------|
| [DINOv2](https://github.com/facebookresearch/dinov2) | 通用视觉特征 |
| [SAM2](https://github.com/facebookresearch/segment-anything-2) | 通用分割 |
| [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | 检测+分割一体 |

---

## 各方案适用场景与限制

### 工业质检方案选型指南

| 方案 | 适用场景 | 优点 | 限制 |
|------|----------|------|------|
| **anomalib（无监督）** | 缺陷样本极少（<50张） | 无需缺陷标注，正常样本即可训练 | 对微小缺陷敏感度有限，FP 率较高 |
| **YOLO 有监督检测** | 有足量缺陷标注（每类>100张）| 精度高，速度快，可定位 | 需要大量标注数据 |
| **Grounding DINO** | 新品类快速验证，不想标注 | 零样本，文字描述即可检测 | 精度不稳定，不适合高精度质检 |
| **传统 OpenCV** | 纹理规律、颜色一致的简单缺陷 | 速度极快，可解释，无 GPU 需求 | 不适应光照变化，泛化差 |

### OCR 方案选型

| 方案 | 适用场景 | 限制 |
|------|----------|------|
| PaddleOCR | 中文工业场景（铭牌/条码/发票）| 英文效果稍弱 |
| EasyOCR | 多语言场景，快速原型 | 速度较慢，精度略低于 PaddleOCR |
| Tesseract | 离线部署，无 GPU | 对倾斜/低质量图像效果差 |

---

## 快速上手：anomalib 工业缺陷检测

以检测某工厂金属螺丝为例，从数据准备到推理完整流程：

### 步骤一：准备数据

```
my_dataset/
├── train/
│   └── good/           # 只有正常样本（200张）
│       ├── 000.jpg
│       └── ...
└── test/
    ├── good/           # 正常测试样本（50张）
    ├── scratch/        # 划痕缺陷（20张）
    └── deformation/    # 变形缺陷（20张）
```

### 步骤二：训练（PatchCore 算法，5 分钟内完成）

```python
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

# 数据模块（Folder 支持自定义目录结构）
datamodule = Folder(
    name="my_screw",
    root="my_dataset",
    normal_dir="train/good",
    abnormal_dir="test",
    normal_test_dir="test/good",
    image_size=256,
    train_batch_size=32,
)

# 模型（PatchCore 是工业场景综合表现最好的算法）
model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,   # 保留 10% 核心特征，控制内存
)

# 训练（PatchCore 无梯度更新，只需提取特征）
engine = Engine(accelerator="gpu", devices=1)
engine.fit(model=model, datamodule=datamodule)

# 评估
results = engine.test(model=model, datamodule=datamodule)
print(f"Image-AUROC: {results[0]['image_AUROC']:.3f}")
```

### 步骤三：单张图像推理

```python
from anomalib.deploy import OpenVINOInferencer  # 或 TorchInferencer
import cv2

# 加载训练好的模型
inferencer = OpenVINOInferencer(
    path="results/patchcore/my_screw/run/weights/openvino/model.bin",
    metadata="results/patchcore/my_screw/run/weights/openvino/metadata.json",
    device="GPU",  # 或 "CPU"
)

# 推理
image = cv2.imread("test_screw.jpg")
result = inferencer.predict(image=image)

print(f"异常分数: {result.pred_score:.4f}")
print(f"判断: {'缺陷' if result.pred_label else '正常'}")

# 可视化热力图
cv2.imshow("热力图", result.heat_map)
cv2.waitKey(0)
```

---

## 工业视觉项目典型技术栈组合

### 方案一：轻量化 CPU 部署（低成本工业 PC）

```
采集：工业相机（海康 / Basler）+ 固定光源
处理：OpenCV 前处理 + ONNX Runtime CPU 推理
模型：YOLOv8n（检测）或 EfficientAD（异常检测）
后端：FastAPI 服务化 + Redis 队列
硬件：Intel 工控机（OpenVINO 加速）
```

### 方案二：高精度 GPU 部署（精密质检）

```
采集：高分辨率线扫相机 + 多角度光源
处理：CUDA 图像处理 + TensorRT FP16 推理
模型：PatchCore 或有监督检测（MMDetection）
后端：Triton Inference Server + gRPC
硬件：NVIDIA RTX 4090 / A30
监控：Grafana + 自定义异常统计看板
```

### 方案三：边缘端（嵌入式部署）

```
采集：USB 相机 + LED 灯环
处理：OpenCV + ncnn / MNN 推理
模型：MobileNet + 蒸馏量化
硬件：Jetson Orin Nano / Raspberry Pi 4
功耗：< 20W
```

---

## 常见工业视觉难点与解决思路

### 难点一：光照变化

工业现场光照条件多变（换班、季节、LED 老化），导致模型泛化差。

**解决方案**：
- 训练时加入光照增强（albumentations: `RandomBrightness`, `RandomGamma`, `CLAHE`）
- 标准化光源（固定光源类型+角度），从源头减少变化
- 使用 `HSV` 颜色空间或归一化颜色特征，降低对绝对亮度的依赖

```python
import albumentations as A

light_aug = A.Compose([
    A.RandomBrightness(limit=0.3, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.RandomShadow(p=0.2),
])
```

### 难点二：遮挡与重叠目标

产线上物体可能部分遮挡，导致检测漏报。

**解决方案**：
- 使用实例分割（Mask R-CNN）代替纯检测
- 训练数据中加入合成遮挡增强（CutOut / CutMix）
- 使用密集预测框架（如 FCOS），减少 anchor 对遮挡的敏感性

### 难点三：小目标（微小缺陷）

瑕疵尺寸可能只有几个像素，标准检测框架难以捕捉。

**解决方案**：
- 提高输入分辨率（1280×1280 代替 640×640）
- 使用切片推理（SAHI 库）：将大图切为小块分别检测后合并

```bash
pip install sahi
```

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='best.pt',
    confidence_threshold=0.3,
    device='cuda:0',
)

result = get_sliced_prediction(
    "large_image.jpg",
    model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,  # 重叠率，避免边界漏检
    overlap_width_ratio=0.2,
)
result.export_visuals("output/")
```

### 难点四：样本不平衡（缺陷率极低）

真实产线缺陷率可能 < 0.1%，难以收集足够缺陷样本。

**解决方案**：
- 无监督异常检测（不需要缺陷标注）
- 缺陷图像合成（将缺陷纹理贴到正常背景）
- 使用 Focal Loss 或权重采样
- 利用 Diffusion 模型生成合成缺陷图像（前沿方向）
