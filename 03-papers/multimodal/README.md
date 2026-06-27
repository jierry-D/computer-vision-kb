# 多模态视觉论文

> 视觉+语言，或视觉基础大模型。

## 视觉-语言预训练

| 论文 | 年份 | 贡献 |
|------|------|------|
| CLIP | 2021 | 对比学习，图文对齐，zero-shot 分类 |
| ALIGN | 2021 | 大规模噪声数据训练 |
| BLIP / BLIP-2 | 2022-2023 | 统一理解+生成 |
| LLaVA | 2023 | 视觉指令微调 |

## 开放词汇检测/分割

| 论文 | 年份 | 贡献 |
|------|------|------|
| GLIP | 2022 | 语言引导目标检测 |
| Grounding DINO | 2023 | 开放词汇检测 SOTA |
| SAM | 2023 | 通用分割大模型 |
| SAM2 | 2024 | 视频+图像统一分割 |
| DINO v2 | 2023 | 自监督视觉特征，通用基础模型 |

## 视觉大模型

| 论文 | 贡献 |
|------|------|
| GPT-4V | 多模态 LLM |
| InternVL | 开源多模态，中文强 |
| Qwen-VL | 阿里多模态 |

---

## CLIP 详解

### 训练目标：对比学习损失

CLIP 使用图像-文本对比学习（InfoNCE Loss），核心思想是让匹配的图文对相似度高、不匹配的低。

对于一个 batch（N 个图文对）：

```
相似度矩阵 logits_per_image[i][j] = cos_sim(image_feat_i, text_feat_j) * temperature

损失 = (CE(行方向) + CE(列方向)) / 2
```

```python
import torch
import torch.nn.functional as F

def clip_loss(image_features, text_features, temperature=0.07):
    """
    image_features: [N, D]，已 L2 归一化
    text_features:  [N, D]，已 L2 归一化
    """
    # 计算相似度矩阵
    logits = torch.matmul(image_features, text_features.T) / temperature  # [N, N]

    # 对角线为正样本对
    labels = torch.arange(len(logits), device=logits.device)

    # 双向对比损失
    loss_i = F.cross_entropy(logits, labels)       # 图->文
    loss_t = F.cross_entropy(logits.T, labels)     # 文->图
    return (loss_i + loss_t) / 2
```

### 零样本分类推理代码（HuggingFace transformers）

```python
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 准备图像和候选类别文本
image = Image.open("cat.jpg")
candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

# 处理输入
inputs = processor(
    text=candidate_labels,
    images=image,
    return_tensors="pt",
    padding=True
)

# 推理
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # [1, 3]
    probs = logits_per_image.softmax(dim=-1)

for label, prob in zip(candidate_labels, probs[0]):
    print(f"{label}: {prob:.3f}")
```

💡 **Tips**：CLIP 的零样本能力强弱与 prompt 写法密切相关。通常 `"a photo of a {class}"` 比直接用类名效果更好。可以用 ensemble（多个 prompt 取平均）进一步提升准确率。

---

## SAM（Segment Anything Model）详解

SAM 支持三种交互式分割方式：

### 方式一：点提示（Point Prompt）

```python
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2

# 加载模型
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# 点提示：点击目标中心
input_point = np.array([[500, 375]])   # [x, y]
input_label = np.array([1])            # 1=前景，0=背景

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,    # 返回3个候选掩码
)
# masks: [3, H, W], scores: [3]，取 scores.argmax() 的掩码
```

### 方式二：框提示（Box Prompt）

```python
# 框提示：给定 bounding box
input_box = np.array([425, 600, 700, 875])  # [x1, y1, x2, y2]，xyxy 格式

masks, scores, logits = predictor.predict(
    box=input_box[None, :],   # 需要加 batch 维
    multimask_output=False,
)
# 框提示通常 multimask_output=False，返回单个最佳掩码
```

### 方式三：自动分割（Automatic Mask Generation）

```python
from segment_anything import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,         # 网格点密度，越大越细致，越慢
    pred_iou_thresh=0.88,       # IoU 置信度阈值
    stability_score_thresh=0.95,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,   # 过滤小区域
)

masks = mask_generator.generate(image_rgb)
# masks 是 list，每个元素包含 'segmentation', 'area', 'bbox', 'stability_score' 等
print(f"共生成 {len(masks)} 个掩码")
```

⚠️ **注意**：自动分割模式在高分辨率图上非常慢（分钟级），工业场景建议用点/框提示模式。

---

## Grounding DINO 使用代码示例

Grounding DINO 支持用自然语言描述来检测任意类别目标（开放词汇检测）。

```python
from groundingdino.util.inference import load_model, load_image, predict

# 加载模型
model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)

# 加载图像
image_source, image = load_image("image.jpg")

# 定义要检测的类别（用 "." 分隔多个类别）
TEXT_PROMPT = "cat . dog . person ."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# 推理
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)

print(f"检测到 {len(boxes)} 个目标")
for box, score, phrase in zip(boxes, logits, phrases):
    print(f"类别: {phrase}, 置信度: {score:.3f}, 框: {box.tolist()}")
```

💡 **Tips**：Grounding DINO 非常适合**免标注**的工业质检场景——只需用文字描述缺陷类型即可，无需准备标注数据。

---

## 多模态模型能力边界

### 能做什么

| 能力 | 说明 | 代表模型 |
|------|------|----------|
| 图像分类（零样本） | 对任意类别进行分类，无需微调 | CLIP |
| 开放词汇检测 | 检测任意文字描述的目标 | Grounding DINO, YOLO-World |
| 通用分割 | 任意目标的像素级分割 | SAM, SAM2 |
| 图像描述生成 | 自动为图像生成文字描述 | BLIP-2, LLaVA |
| 视觉问答（VQA） | 回答关于图像内容的问题 | GPT-4V, InternVL, Qwen-VL |
| 文生图 | 根据文字描述生成图像 | Stable Diffusion, DALL-E 3 |
| 视频理解 | 视频内容描述和问答 | SAM2, VideoLLaMA |

### 不能（或不擅长）做什么

| 局限 | 说明 |
|------|------|
| 精确像素级度量 | 无法给出精确尺寸、角度数值 |
| 工业级缺陷检出率 | 零样本检测漏检率较高，不适合高精度质检 |
| 实时处理 | 大多数 VLM 推理速度慢，不适合视频流处理 |
| 领域专属细粒度识别 | 对专业领域（如特定型号零件区分）效果差 |
| 精确计数 | 密集目标计数准确率低 |

---

## VLM 评估基准

| 基准 | 全称 | 评估内容 | 说明 |
|------|------|----------|------|
| MMMU | Massive Multidisciplinary Multimodal Understanding | 跨学科多图问答 | 难度高，综合能力测试 |
| MMBench | - | 多维度 VQA | 中英文双版本，覆盖面广 |
| SEEDBench | - | 多粒度图像理解 | 包含视频理解子集 |
| MM-Vet | - | 视觉对话综合能力 | 复杂推理链测试 |
| HallusionBench | - | 视觉幻觉评估 | 测试模型是否"看图说瞎话" |
| TextVQA | - | 图中文字理解 | OCR 相关 |
| ChartQA | - | 图表理解和推理 | 数据图表问答 |

💡 **Tips**：HallusionBench 是目前最重要的评估之一——很多 VLM 会对图像中不存在的内容"自信地"描述，这在工业场景中是严重问题。选型时务必关注幻觉率。
