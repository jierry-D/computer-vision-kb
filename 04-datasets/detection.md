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
