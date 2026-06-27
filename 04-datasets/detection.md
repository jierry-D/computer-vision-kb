# 目标检测数据集

| 数据集 | 规模 | 类别 | 特点 |
|--------|------|------|------|
| PASCAL VOC 2012 | 11k 图 | 20 类 | 经典基准，入门首选 |
| MS COCO | 330k 图 | 80 类 | 当前主流基准，多任务 |
| OpenImages V7 | 9M 图 | 600 类 | 最大规模，长尾分布 |
| Objects365 | 2M 图 | 365 类 | 大规模预训练数据 |
| VisDrone | 无人机视角 | 10 类 | 小目标，密集场景 |
| KITTI | 自动驾驶 | 3 类 | 车辆/行人/骑行者 |
| nuScenes | 自动驾驶 | 23 类 | 多传感器，3D 检测 |

## COCO 评估指标说明

```
AP     = mAP@[0.5:0.05:0.95]  # COCO 标准主指标
AP50   = mAP@IoU=0.50
AP75   = mAP@IoU=0.75
APs    = small objects  (<32²px)
APm    = medium objects (32²~96²px)
APl    = large objects  (>96²px)
```

## 下载工具

```bash
# COCO
pip install fiftyone
import fiftyone.zoo as foz
dataset = foz.load_zoo_dataset("coco-2017", split="train")
```

---

## COCO 数据集目录结构

```
coco/
├── annotations/
│   ├── instances_train2017.json    # 目标检测+实例分割标注
│   ├── instances_val2017.json
│   ├── captions_train2017.json     # 图像描述标注
│   ├── captions_val2017.json
│   ├── person_keypoints_train2017.json  # 关键点标注
│   └── person_keypoints_val2017.json
├── train2017/
│   ├── 000000000009.jpg
│   └── ...（118287 张）
├── val2017/
│   ├── 000000000139.jpg
│   └── ...（5000 张）
└── test2017/
    └── ...（40775 张，无标注）
```

### annotations JSON 格式

```json
{
  "info": {...},
  "licenses": [...],
  "images": [
    {
      "id": 9,
      "file_name": "000000000009.jpg",
      "height": 480, "width": 640,
      "date_captured": "2013-11-15 02:41:42"
    }
  ],
  "annotations": [
    {
      "id": 1038,
      "image_id": 9,
      "category_id": 1,
      "segmentation": [[...多边形点坐标...]],
      "area": 702.1057499999998,
      "bbox": [473.07, 395.93, 38.65, 28.67],  // [x, y, w, h] 格式
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "person", "supercategory": "person"},
    ...
  ]
}
```

⚠️ **注意**：COCO 的 bbox 格式是 `[x_min, y_min, width, height]`（XYWH），而非 `[x_min, y_min, x_max, y_max]`（XYXY）。许多框架内部使用 XYXY 格式，注意转换。

---

## COCO API 使用代码

```python
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 加载标注
ann_file = "coco/annotations/instances_val2017.json"
coco = COCO(ann_file)

# 获取所有类别
cats = coco.loadCats(coco.getCatIds())
cat_names = [c['name'] for c in cats]
print(f"共 {len(cat_names)} 个类别: {cat_names[:10]}...")

# 按类别获取图像
cat_ids = coco.getCatIds(catNms=['person', 'car'])
img_ids = coco.getImgIds(catIds=cat_ids)
print(f"包含 person 或 car 的图像: {len(img_ids)} 张")

# 加载并可视化标注
img_id = img_ids[0]
img_info = coco.loadImgs(img_id)[0]

# 加载图像
image = Image.open(f"coco/val2017/{img_info['file_name']}")

# 获取该图的所有标注
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)

# 可视化
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)
coco.showAnns(anns)   # 画掩码
for ann in anns:
    x, y, w, h = ann['bbox']
    rect = patches.Rectangle((x, y), w, h, linewidth=2,
                               edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    cat_name = coco.loadCats(ann['category_id'])[0]['name']
    ax.text(x, y - 5, cat_name, color='red', fontsize=10)
plt.axis('off')
plt.show()
```

---

## VOC 和 COCO 格式互转

### VOC XML 格式说明

```xml
<annotation>
  <filename>2007_000032.jpg</filename>
  <size>
    <width>500</width><height>281</height>
  </size>
  <object>
    <name>aeroplane</name>
    <bndbox>
      <xmin>104</xmin><ymin>78</ymin>
      <xmax>375</xmax><ymax>183</ymax>
    </bndbox>
    <difficult>0</difficult>
  </object>
</annotation>
```

### VOC 转 COCO 格式脚本

```python
import os, json, xml.etree.ElementTree as ET
from pathlib import Path

def voc_to_coco(voc_dir, output_json):
    """将 VOC 格式标注转换为 COCO JSON 格式"""
    categories = [
        {"id": 1, "name": "aeroplane"},
        {"id": 2, "name": "bicycle"},
        # ... 所有类别
    ]
    cat_name2id = {c['name']: c['id'] for c in categories}

    images, annotations = [], []
    ann_id = 1

    xml_files = sorted(Path(voc_dir).glob("*.xml"))
    for img_id, xml_file in enumerate(xml_files, start=1):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        images.append({"id": img_id, "file_name": filename,
                        "width": w, "height": h})

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in cat_name2id:
                continue
            bbox_elem = obj.find('bndbox')
            xmin = float(bbox_elem.find('xmin').text)
            ymin = float(bbox_elem.find('ymin').text)
            xmax = float(bbox_elem.find('xmax').text)
            ymax = float(bbox_elem.find('ymax').text)
            # COCO 格式：[x, y, width, height]
            coco_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            area = (xmax - xmin) * (ymax - ymin)

            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": cat_name2id[name],
                "bbox": coco_bbox, "area": area,
                "iscrowd": 0
            })
            ann_id += 1

    coco_data = {"images": images, "annotations": annotations,
                 "categories": categories}
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    print(f"转换完成: {len(images)} 张图, {len(annotations)} 个标注")

# 使用
voc_to_coco("VOCdevkit/VOC2012/Annotations", "coco_train.json")
```

---

## 自定义数据集转 COCO 格式脚本（通用版）

```python
import json
from pathlib import Path
from PIL import Image

def build_coco_dataset(image_dir, label_dir, class_names, output_json):
    """
    通用自定义数据集转 COCO 格式
    假设标注格式：每张图一个 txt 文件，每行 "class_id x_center y_center w h"（YOLO 格式，归一化）
    """
    images, annotations, categories = [], [], []
    categories = [{"id": i+1, "name": name}
                  for i, name in enumerate(class_names)]

    ann_id = 1
    for img_id, img_path in enumerate(sorted(Path(image_dir).glob("*.jpg")), start=1):
        img = Image.open(img_path)
        img_w, img_h = img.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w, "height": img_h
        })

        # 读取对应 YOLO 格式标注
        label_path = Path(label_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                cls_id, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
                # 转换为绝对坐标 XYWH
                x = (xc - bw / 2) * img_w
                y = (yc - bh / 2) * img_h
                w = bw * img_w
                h = bh * img_h
                annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cls_id + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h, "iscrowd": 0
                })
                ann_id += 1

    result = {"images": images, "annotations": annotations,
              "categories": categories}
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"生成 {len(images)} 张图, {len(annotations)} 个标注")

build_coco_dataset(
    "dataset/images", "dataset/labels",
    class_names=["cat", "dog", "person"],
    output_json="dataset/train.json"
)
```

---

## 数据集统计分析代码

```python
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_coco_dataset(ann_file):
    """分析 COCO 格式数据集的统计信息"""
    with open(ann_file) as f:
        data = json.load(f)

    cat_id2name = {c['id']: c['name'] for c in data['categories']}

    # 1. 类别分布
    cat_counts = Counter(ann['category_id'] for ann in data['annotations'])
    sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    names = [cat_id2name[cid] for cid, _ in sorted_cats]
    counts = [cnt for _, cnt in sorted_cats]
    plt.barh(names[:20], counts[:20])
    plt.title("类别分布（Top 20）")
    plt.xlabel("标注数量")

    # 2. 目标尺寸分布
    plt.subplot(1, 2, 2)
    areas = np.array([ann['area'] for ann in data['annotations']])
    widths  = np.array([ann['bbox'][2] for ann in data['annotations']])
    heights = np.array([ann['bbox'][3] for ann in data['annotations']])

    # 按尺寸分类（参考 COCO 标准）
    small  = (areas < 32**2).sum()
    medium = ((areas >= 32**2) & (areas < 96**2)).sum()
    large  = (areas >= 96**2).sum()
    plt.pie([small, medium, large],
            labels=[f"小目标\n(<32²)\n{small}", f"中目标\n(32²-96²)\n{medium}",
                    f"大目标\n(>96²)\n{large}"],
            autopct='%1.1f%%')
    plt.title("目标尺寸分布")

    plt.tight_layout()
    plt.savefig("dataset_analysis.png", dpi=150)
    plt.show()

    print(f"\n数据集统计：")
    print(f"  图像数: {len(data['images'])}")
    print(f"  标注数: {len(data['annotations'])}")
    print(f"  类别数: {len(data['categories'])}")
    print(f"  平均每图标注数: {len(data['annotations'])/len(data['images']):.1f}")
    print(f"  目标尺寸分布: 小={small} 中={medium} 大={large}")

analyze_coco_dataset("coco/annotations/instances_val2017.json")
```
