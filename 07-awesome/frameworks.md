# 精选训练框架

## 通用检测/分割框架

| 框架 | Stars | 特点 |
|------|-------|------|
| [MMDetection](https://github.com/open-mmlab/mmdetection) | 28k+ | OpenMMLab，模型最全 |
| [Detectron2](https://github.com/facebookresearch/detectron2) | 30k+ | Meta，研究友好 |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | 35k+ | YOLOv8/v11，工业易用 |
| [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) | 12k+ | 百度，PP-YOLOE/RT-DETR |

## 分割

| 框架 | 特点 |
|------|------|
| [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) | 语义分割，模型覆盖全 |
| [Mask2Former](https://github.com/facebookresearch/Mask2Former) | 统一分割框架 |
| [SAM2](https://github.com/facebookresearch/segment-anything-2) | Meta 通用分割 |

## 三维视觉

| 框架 | 特点 |
|------|------|
| [Open3D](https://github.com/isl-org/Open3D) | 点云/网格处理标准库 |
| [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) | NeRF/3DGS 统一平台 |
| [COLMAP](https://github.com/colmap/colmap) | SfM 标准工具 |

## 异常检测

| 框架 | 特点 |
|------|------|
| [anomalib](https://github.com/openvinotoolkit/anomalib) | Intel，工业异常检测大合集 |

---

## 安装命令

```bash
# Ultralytics YOLO（最推荐入门）
pip install ultralytics

# MMDetection（功能最全）
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" mmdet

# Detectron2（Meta 研究版本，安装较复杂）
pip install torch torchvision
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# anomalib（工业异常检测）
pip install anomalib

# Open3D（点云处理）
pip install open3d

# SAM2（通用分割）
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

---

## 快速上手示例：用 Ultralytics YOLO 检测一张图

```python
from ultralytics import YOLO
import cv2

# 加载预训练模型（自动下载权重）
model = YOLO("yolov8n.pt")          # nano，最快
# model = YOLO("yolov8s.pt")        # small，平衡
# model = YOLO("yolov8x.pt")        # extra large，最准

# 检测一张图像
results = model("image.jpg")

# 访问结果
for result in results:
    boxes = result.boxes             # 检测框
    masks = result.masks             # 分割掩码（需用分割模型）

    # 打印检测结果
    for box in boxes:
        xyxy  = box.xyxy[0].tolist()    # [x1, y1, x2, y2]
        conf  = box.conf[0].item()       # 置信度
        cls   = int(box.cls[0].item())   # 类别 ID
        label = model.names[cls]         # 类别名
        print(f"{label}: conf={conf:.3f}, box={[round(x) for x in xyxy]}")

    # 保存可视化结果
    result.save("result.jpg")

# 批量推理（更高效）
results = model(["img1.jpg", "img2.jpg", "img3.jpg"], batch=8)

# 视频推理
results = model("video.mp4", stream=True)  # stream=True 节省内存
for result in results:
    frame = result.plot()   # 绘制结果到帧
    cv2.imshow("YOLO", frame)
    if cv2.waitKey(1) == ord('q'):
        break
```

### 训练自定义数据集

```bash
# 1. 准备 YOLO 格式数据集
# dataset/
# ├── images/train/
# ├── images/val/
# ├── labels/train/    # 每行：class_id x_center y_center w h（归一化）
# └── labels/val/

# 2. 创建 data.yaml
cat > data.yaml << EOF
path: ./dataset
train: images/train
val: images/val
nc: 3
names: ['cat', 'dog', 'person']
EOF

# 3. 训练
yolo detect train \
    data=data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0
```

---

## 框架选型建议

### 学术研究 → MMDetection / Detectron2

```
优先选 MMDetection（中文文档，社区大）或 Detectron2（Meta 维护，与论文代码关联紧密）

理由：
- 模型覆盖最全，SOTA 论文通常提供这两个框架的实现
- 配置文件系统便于消融实验
- 方便与论文结果对比复现

缺点：上手成本较高，初次配置需要 1-2 天
```

### 工业落地 → Ultralytics YOLO

```
优先选 Ultralytics YOLO（特别是 YOLOv8/YOLOv11）

理由：
- 文档完善，5 分钟上手
- 原生支持导出 ONNX/TRT/CoreML/TFLite 等格式
- 精度和速度在工业场景有良好平衡
- 活跃的社区和频繁的更新

缺点：定制化自由度不如 MMDetection
```

### 快速原型验证 → Ultralytics YOLO + Roboflow

```
适合：有想法但想快速验证可行性

流程：
1. Roboflow 标注数据（Web 端，无需本地环境）
2. Roboflow 导出 YOLO 格式
3. Ultralytics 5 行代码开始训练
4. 几小时内看到结果
```

### 异常检测 → anomalib

```
适合：工业质检，无（少）缺陷样本场景

内置算法：PatchCore, PaDiM, EfficientAD, FastFlow, CFA 等 30+ 种
从 MVTec 到自定义数据集只需改几行配置
```

---

## 社区活跃度与维护状态

| 框架 | 维护方 | 更新频率 | 社区 | 建议 |
|------|--------|----------|------|------|
| Ultralytics YOLO | Ultralytics 公司 | 非常活跃（每周更新） | 极大 | 首选工业项目 |
| MMDetection | 上海 AI Lab + OpenMMLab 社区 | 活跃（月度更新） | 大，中文活跃 | 首选学术研究 |
| Detectron2 | Meta AI | 中等（季度更新） | 中，英文 | Meta 论文配套 |
| anomalib | Intel OpenVINO 团队 | 活跃（月度更新） | 中等 | 工业异常检测首选 |
| PaddleDetection | 百度 | 活跃 | 中，中文 | 国内工业部署（配套 PaddleLite） |
| SAM2 | Meta AI | 较活跃 | 大 | 通用分割基础模型 |
| nerfstudio | 开源社区 | 活跃 | 中等 | NeRF/3DGS 研究 |

💡 **Tips**：框架选型原则：**优先选代码质量高、文档完善、社区活跃的框架**，而非最新或 Star 最多的。一个框架的 issue 回复速度往往能反映其维护质量。
