# 工业视觉数据集

## 缺陷检测

| 数据集 | 规模 | 类别 | 特点 |
|--------|------|------|------|
| MVTec AD | 5354 图 | 15 类 | 异常检测标准基准 |
| VisA | 10821 图 | 12 类 | 高分辨率，多种品类 |
| MPDD | 1346 图 | 6 类 | 金属零件缺陷 |
| DAGM | 6900 图 | 10 类 | 纹理类缺陷 |
| NEU-DET | 1800 图 | 6 类 | 钢材表面缺陷 |
| SDNET2018 | 56k 图 | 2 类 | 混凝土裂缝 |

## 工业测量 / OCR

| 数据集 | 用途 |
|--------|------|
| IC15/IC17 | 自然场景文字检测 |
| SVHN | 街景门牌号码 |
| MSRA-TD500 | 多方向文字 |

## 注意事项

- MVTec AD 只提供正常样本用于训练，测试集含缺陷（无监督异常检测设定）
- 工业场景建议先在 MVTec 跑通，再迁移到实际数据

---

## MVTec AD 各类别缺陷类型详解

MVTec AD 包含 15 种工业品类，每类有特定的缺陷模式：

| 品类 | 类型 | 典型缺陷 |
|------|------|----------|
| Bottle | 纹理+结构 | 破损（broken_large/small）、污染（contamination） |
| Cable | 结构 | 弯曲、缺失、损坏的芯线 |
| Capsule | 结构 | 裂纹、挤压变形、缺失印刷、污点 |
| Carpet | 纹理 | 颜色变异、切割、孔洞、金属污染 |
| Grid | 纹理 | 弯曲、断裂、胶水、金属污染 |
| Hazelnut | 结构 | 裂缝、切割、孔洞、印刷错误 |
| Leather | 纹理 | 颜色变异、切割、折叠、粘合剂、孔洞 |
| Metal Nut | 结构 | 弯曲、颜色变异、翻转、划痕 |
| Pill | 结构 | 颜色变异、合并、污染、裂纹、缺口、划痕、类型错误 |
| Screw | 结构 | 螺纹损坏、操纵划痕、划痕头部/侧面 |
| Tile | 纹理 | 裂纹、胶水条纹、灰度变化、油污、粗糙表面 |
| Toothbrush | 结构 | 缺陷刷毛（仅 1 种缺陷类型）|
| Transistor | 结构 | 弯曲引脚、损坏外壳、错误放置 |
| Wood | 纹理 | 颜色变化、组合缺陷、孔洞、液体、划痕 |
| Zipper | 结构 | 布料缺陷、损坏齿、拉链上移、误对齐、孔洞、粗糙 |

💡 **Tips**：MVTec AD 的缺陷类别设计体现了两大类工业缺陷：**纹理类**（Ground texture, Carpet, Leather 等）和**结构类**（Cable, Screw, Transistor 等）。不同类别适合测试异常检测算法对不同类型异常的处理能力。

---

## anomalib 加载 MVTec 代码示例

anomalib 是目前最完整的工业异常检测框架，内置了 30+ 种算法。

```bash
# 安装
pip install anomalib
```

```python
from anomalib.data import MVTec
from anomalib.models import Padim    # 常用方法：PaDiM, PatchCore, EfficientAD
from anomalib.engine import Engine
from anomalib.loggers import AnomalibWandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# 数据配置
datamodule = MVTec(
    root="datasets/MVTec",    # MVTec AD 数据集根目录
    category="bottle",        # 选择品类
    image_size=256,
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
)

# 模型
model = Padim(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
)

# 训练引擎
engine = Engine(
    max_epochs=1,          # PaDiM 一般只需 1 epoch（提取特征）
    accelerator="gpu",
    devices=1,
)

# 训练
engine.fit(model=model, datamodule=datamodule)

# 评估
engine.test(model=model, datamodule=datamodule)
```

```python
# 批量测试 MVTec 所有品类
from anomalib.data import MVTec
from anomalib.models import PatchCore
from anomalib.engine import Engine

categories = ["bottle", "cable", "capsule", "carpet", "grid",
              "hazelnut", "leather", "metal_nut", "pill", "screw",
              "tile", "toothbrush", "transistor", "wood", "zipper"]

results = {}
for category in categories:
    datamodule = MVTec(root="datasets/MVTec", category=category, image_size=224)
    model = PatchCore(backbone="wide_resnet50_2", coreset_sampling_ratio=0.1)
    engine = Engine(max_epochs=1, accelerator="gpu", devices=1)
    engine.fit(model=model, datamodule=datamodule)
    metrics = engine.test(model=model, datamodule=datamodule)
    results[category] = metrics
    print(f"{category}: Image-AUROC = {metrics[0]['image_AUROC']:.3f}")
```

---

## 工业数据标注工具对比

| 工具 | 适用场景 | 特点 | 推荐指数 |
|------|----------|------|----------|
| CVAT | 工业质检，视频流 | 开源、功能全、支持半自动（SAM 辅助） | ★★★★★ |
| Label Studio | 多任务团队协作 | 灵活的 JSON 配置，REST API，支持 ML backend | ★★★★ |
| Roboflow | 快速原型 | 云端一站式，内置增强和格式转换 | ★★★★ |
| LabelMe | 小规模自建数据 | 轻量，本地，Python 友好 | ★★★ |
| VGG Image Annotator (VIA) | 临时标注 | 纯 HTML，无需安装 | ★★ |

### CVAT 快速部署（Docker）

```bash
git clone https://github.com/opencv/cvat
cd cvat
docker compose up -d
# 访问 http://localhost:8080
```

### 工业标注最佳实践

1. **标注规范先行**：制定明确的标注指南（缺陷边界在哪里算？轻微缺陷算不算？），保证标注一致性
2. **先标一批，评估质量**：标 100 张，两人交叉验证，统计一致性（IOU/Kappa）
3. **分批标注**：不要一次性标完，训练后用模型辅助标注（AL）提高效率
4. **正常样本也要标注质量**：工业无监督检测中，训练集的"正常样本纯净度"至关重要

---

## 如何用正常样本构建训练集

工业无监督异常检测的关键是**构建高质量的正常样本训练集**：

### 数据采集原则

| 要素 | 建议 |
|------|------|
| 数量 | 每个品类至少 200 张，建议 500-1000 张 |
| 光照 | 固定光源（环形光/背光），避免阴影和反光变化 |
| 相机角度 | 固定支架，消除视角变化 |
| 样本多样性 | 覆盖合法的外观变化（颜色批次差异/正常纹理变化） |
| 纯净度 | 仔细检查，确保没有缺陷样本混入训练集 |

### 数据质量检查

```python
import os
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_outliers_in_normal_set(image_dir, n_clusters=5):
    """
    用特征聚类找出训练集中可能的异常样本
    （混入缺陷样本会导致训练集不纯净）
    """
    # 提取特征
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    features, filenames = [], []
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        img = Image.open(os.path.join(image_dir, fname)).convert('RGB')
        with torch.no_grad():
            feat = model(transform(img).unsqueeze(0)).squeeze().numpy()
        features.append(feat)
        filenames.append(fname)

    features = np.array(features)

    # 计算每个样本到聚类中心的距离
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
    distances = np.min(kmeans.transform(features), axis=1)

    # 找出距离最远的样本（潜在异常）
    threshold = np.percentile(distances, 95)
    outliers = [(filenames[i], distances[i])
                for i in range(len(filenames)) if distances[i] > threshold]
    print(f"可能的异常样本（前 5%）：")
    for fname, dist in sorted(outliers, key=lambda x: -x[1])[:10]:
        print(f"  {fname}: 距离 = {dist:.4f}")
```

---

## 工业场景数据采集建议

### 光源选择

| 光源类型 | 适用缺陷 | 效果 |
|----------|----------|------|
| 环形光（Ring Light） | 表面划痕、凹坑 | 突出反光差异 |
| 背光（Backlight） | 孔洞、尺寸测量 | 剪影效果，边缘清晰 |
| 同轴光（Coaxial Light） | 印刷/平面缺陷 | 消除镜面反光 |
| 线扫描光 | 高速传送带检测 | 适合运动物体 |
| 多角度联合 | 复杂缺陷 | 多光源组合揭示不同缺陷 |

### 相机与镜头建议

- **工业相机**：Basler/海康 MV 系列，全局快门（避免运动模糊）
- **分辨率**：最小缺陷尺寸 ≥ 3×3 像素，建议 10 像素以上
- **镜头**：远心镜头可消除透视变形，适合精密测量
- **帧率**：传送带速度 / 视野宽度 = 最低帧率

### 采集数量建议

| 场景 | 正常样本（训练） | 缺陷样本（测试/验证） |
|------|-----------------|----------------------|
| 无监督异常检测（PatchCore等） | ≥ 200，建议 500+ | 每类缺陷 20-50 张 |
| 有监督缺陷检测（YOLO等） | 与缺陷比约 1:1 | 每类 ≥ 100 张 |
| 精度高要求 | ≥ 1000 | 尽量多 |

⚠️ **注意**：工业现场真实缺陷样本通常稀缺，常见策略是：用正常样本+图像增强（划痕合成、纹理贴图）生成合成缺陷数据，配合真实少量缺陷数据训练。
