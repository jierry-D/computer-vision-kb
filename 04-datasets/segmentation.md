# 分割数据集

## 语义分割

| 数据集 | 规模 | 类别 | 场景 |
|--------|------|------|------|
| PASCAL VOC | 11k | 21 | 通用 |
| ADE20K | 25k | 150 | 室内外通用，细粒度 |
| Cityscapes | 5k 精标 | 19 | 自动驾驶街景 |
| COCO-Stuff | 164k | 172 | stuff+things |
| BDD100K | 100k | 19 | 自动驾驶，多任务 |

## 实例分割

| 数据集 | 规模 | 特点 |
|--------|------|------|
| COCO Instance | 330k | 主流基准 |
| LVIS | 164k | 长尾，1203 类 |
| SBD | 11k | Pascal 扩展 |

## 医学图像分割

| 数据集 | 器官/病灶 | 模态 |
|--------|---------|------|
| ISIC | 皮肤病变 | 皮肤镜 |
| KiTS | 肾脏肿瘤 | CT |
| BraTS | 脑肿瘤 | MRI |
| DRIVE | 视网膜血管 | 眼底图 |

---

## ADE20K 颜色编码规则

ADE20K 提供两种标注格式，需要理解颜色与类别的对应关系：

### 格式一：RGB 编码（推荐）

在官方 `_seg.png` 文件中，颜色按以下规则映射类别 ID：

```python
import numpy as np
from PIL import Image

def decode_ade20k_label(seg_path):
    """
    ADE20K _seg.png 文件解码
    类别 ID = R/10 * 256 + G，实例 ID = B
    """
    seg = np.array(Image.open(seg_path))
    # R 通道包含类别信息高位，G 通道为低位
    class_mask = (seg[:, :, 0] / 10).astype(np.uint8) * 256 + seg[:, :, 1]
    instance_mask = seg[:, :, 2]
    return class_mask, instance_mask

class_mask, instance_mask = decode_ade20k_label("ADE/ADEChallengeData2016/annotations/training/ADE_train_00000001_seg.png")
print(f"类别范围: {class_mask.min()} ~ {class_mask.max()}")  # 0=背景, 1~150=类别
```

### 格式二：JSON 标注（详细多边形）

ADE20K 也提供 JSON 文件包含详细的多边形标注，适合实例分割使用。

### 150 个类别概览

ADE20K 的 150 个类别覆盖室内外场景，按频次排序前 10 为：
`wall, building, sky, floor, tree, ceiling, road, bed, windowpane, grass`

💡 **Tips**：ADE20K 的类别 0 表示背景（无物体区域），训练时通常忽略背景损失（`ignore_index=0`）。

---

## Cityscapes 标注层级

Cityscapes 有两种标注精度，适用场景不同：

### 精细标注（Fine Annotation）

- **规模**：5000 张（train: 2975 / val: 500 / test: 1525）
- **标注质量**：人工精细描绘多边形，每张图标注平均耗时 1.5 小时
- **用途**：训练和评估语义/实例分割模型，比赛提交

### 粗糙标注（Coarse Annotation）

- **规模**：20000 张（另外 20k 张弱标注）
- **标注质量**：粗略多边形，质量较低
- **用途**：与精细标注联合训练，提升模型泛化

### 19 个训练类别与颜色编码

```python
# Cityscapes 标准颜色映射（RGB）
CITYSCAPES_COLORS = {
    0:  (128, 64, 128),   # road
    1:  (244, 35, 232),   # sidewalk
    2:  (70, 70, 70),     # building
    3:  (102, 102, 156),  # wall
    4:  (190, 153, 153),  # fence
    5:  (153, 153, 153),  # pole
    6:  (250, 170, 30),   # traffic light
    7:  (220, 220, 0),    # traffic sign
    8:  (107, 142, 35),   # vegetation
    9:  (152, 251, 152),  # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
    255: (0, 0, 0),       # ignore（void）
}

def colorize_segmentation(pred_mask, color_map):
    """将语义分割预测图着色"""
    h, w = pred_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        color_img[pred_mask == class_id] = color
    return color_img
```

⚠️ **注意**：Cityscapes 原始标注包含 34 个类别，但评估标准只用 19 个，标注中 `trainId` 字段才是训练用 ID，`255` 表示忽略区域。

---

## 医学图像分割数据预处理流程

医学图像（CT/MRI）与自然图像差异很大，需要特殊预处理：

```python
import numpy as np
import nibabel as nib   # pip install nibabel
from scipy import ndimage

def preprocess_ct_scan(nifti_path, target_spacing=(1.0, 1.0, 2.0)):
    """CT 扫描标准预处理流程"""
    # 1. 加载 NIfTI 格式
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()
    spacing = nii.header.get_zooms()[:3]  # (x, y, z) 体素间距，mm

    # 2. 重采样到统一分辨率（各向同性）
    zoom_factors = [s / t for s, t in zip(spacing, target_spacing)]
    volume = ndimage.zoom(volume, zoom_factors, order=1)

    # 3. HU 值窗宽窗位（CT 特有，软组织窗：W=400 L=40）
    window_center, window_width = 40, 400
    hu_min = window_center - window_width // 2
    hu_max = window_center + window_width // 2
    volume = np.clip(volume, hu_min, hu_max)

    # 4. 归一化到 [0, 1]
    volume = (volume - hu_min) / (hu_max - hu_min)

    return volume.astype(np.float32)

# MRI 预处理（不用 HU 窗，用 z-score 归一化）
def preprocess_mri(nifti_path):
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()
    # Z-score 归一化（仅对非零区域）
    mask = volume > 0
    mean_val = volume[mask].mean()
    std_val  = volume[mask].std()
    volume[mask] = (volume[mask] - mean_val) / (std_val + 1e-8)
    return volume.astype(np.float32)
```

### 2D 切片提取（适合大多数医学分割任务）

```python
def extract_2d_slices(volume, label, min_nonzero_ratio=0.05):
    """从 3D 体数据提取有效 2D 切片"""
    slices = []
    for i in range(volume.shape[2]):  # 沿 z 轴切片
        slice_label = label[:, :, i]
        # 只保留包含足够前景的切片
        if slice_label.sum() / slice_label.size > min_nonzero_ratio:
            slices.append((volume[:, :, i], slice_label))
    return slices
```

---

## 标注工具推荐（制作语义分割数据）

### LabelMe（推荐，轻量易用）

```bash
# 安装
pip install labelme

# 启动
labelme
# 或指定图像目录
labelme /path/to/images --output /path/to/labels --autosave
```

**LabelMe 生成语义分割掩码**：

```python
import labelme
import json, numpy as np
from PIL import Image

def labelme_json_to_mask(json_path, label2id):
    """将 LabelMe JSON 标注转为语义分割掩码"""
    with open(json_path) as f:
        data = json.load(f)

    height, width = data['imageHeight'], data['imageWidth']
    mask = np.zeros((height, width), dtype=np.uint8)

    for shape in data['shapes']:
        label = shape['label']
        if label not in label2id:
            continue
        pts = np.array(shape['points'], dtype=np.int32)
        # 多边形填充
        from PIL import ImageDraw
        img = Image.fromarray(mask)
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(p) for p in pts], fill=label2id[label])
        mask = np.array(img)

    return mask

label2id = {"background": 0, "crack": 1, "rust": 2}
mask = labelme_json_to_mask("annotation.json", label2id)
```

### 其他推荐标注工具对比

| 工具 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| LabelMe | 小数据集，本地使用 | 轻量，Python，格式简单 | 无团队协作 |
| Label Studio | 团队协作，多任务 | 支持多种标注类型，API 完整 | 部署复杂 |
| CVAT | 视频+图像，工业级 | 功能全，支持半自动 | 较重，需要服务器 |
| Labelbox | 商业项目 | 云端，AI 辅助标注 | 收费 |
| Roboflow | 检测/分割全流程 | 内置增强和格式转换 | 免费版有限制 |

---

## 数据集格式统一工具

不同任务、不同框架使用的格式不统一，以下工具帮助转换：

```bash
# 使用 supervision 库（强烈推荐）
pip install supervision

# 支持 COCO / YOLO / VOC / Pascal 等格式互转
```

```python
import supervision as sv

# 加载 COCO 格式
dataset = sv.DetectionDataset.from_coco(
    images_directory_path="images/",
    annotations_path="annotations.json"
)

# 转为 YOLO 格式
dataset.as_yolo(
    images_directory_path="yolo_dataset/images/",
    annotations_directory_path="yolo_dataset/labels/",
    data_yaml_path="yolo_dataset/data.yaml"
)
```

💡 **Tips**：Roboflow 的在线平台也提供图形化格式转换功能，适合不想写代码的场景，免费版支持 3 个项目。
