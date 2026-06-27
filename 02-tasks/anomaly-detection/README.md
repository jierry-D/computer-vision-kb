# 异常检测（工业质检）

> 在无（少量）缺陷样本标注的情况下，检测图像中的异常区域。

## 方法分类

### 无监督/半监督（主流）

| 方法 | 代表模型 | 特点 |
|------|---------|------|
| 特征嵌入 | PatchCore | 核心集存储正常特征，近邻搜索 |
| 知识蒸馏 | STFPM、RD4AD | 教师-学生特征差异 |
| 标准化流 | FastFlow | 学习正常分布，异常低概率 |
| 重建 | DRAEM、SimpleNet | 重建误差作为异常分数 |

### 有监督

- 直接用少量缺陷样本训练分类/分割模型

## 主流 Benchmark

- **MVTec AD**：15 类工业品，含纹理+物体
- **VisA**：12 类，高分辨率
- **DAGM**：纹理类工业缺陷

## 评估指标

- **AUROC**（图像级/像素级）
- **AP**（像素级）
- **PRO**（Per-Region Overlap）

---

## PatchCore 完整算法步骤

PatchCore 是目前最常用的工业异常检测方案，核心思路：用预训练 CNN 提取正常样本的 patch 特征，构建核心集，推理时用近邻搜索计算异常分数。

### 算法流程

```
训练阶段：
1. 使用预训练 ResNet/WideResNet（ImageNet 权重，不更新）
2. 提取训练集（全为正常样本）所有 patch 的中间层特征
   - 取 layer2 + layer3 的输出，双线性插值到相同空间分辨率后拼接
3. 用 Coreset 采样（贪心稀疏采样）压缩内存库（保留 1%~10%）
4. 保存核心集到内存库 M

推理阶段：
1. 提取测试图像的 patch 特征
2. 对每个 patch，搜索内存库中最近邻的距离（异常分数）
3. 像素级异常图：将 patch 分数上采样回原图大小
4. 图像级分数：取异常图的最大值
```

```python
import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
from sklearn.random_projection import SparseRandomProjection

class PatchCore:
    def __init__(self, backbone='wide_resnet50_2', coreset_ratio=0.01,
                 patch_size=3):
        # 使用预训练骨干（不微调）
        net = getattr(models, backbone)(pretrained=True)
        self.feature_extractor = FeatureExtractor(net, layers=['layer2', 'layer3'])
        self.memory_bank = None
        self.coreset_ratio = coreset_ratio

    def fit(self, dataloader):
        """用正常样本构建内存库"""
        all_features = []
        self.feature_extractor.eval()
        with torch.no_grad():
            for images, _ in dataloader:
                feats = self._extract_patch_features(images)  # [N, D]
                all_features.append(feats)
        all_features = torch.cat(all_features, dim=0)  # [N_total, D]

        # 随机投影降维（加速近邻搜索）
        projector = SparseRandomProjection(n_components='auto', eps=0.9)
        reduced = projector.fit_transform(all_features.cpu().numpy())

        # Greedy Coreset 采样（保留最具代表性的子集）
        n_coreset = max(1, int(len(all_features) * self.coreset_ratio))
        indices = self._greedy_coreset_sample(reduced, n_coreset)
        self.memory_bank = all_features[indices]  # [n_coreset, D]
        print(f"内存库大小: {self.memory_bank.shape}")

    def _greedy_coreset_sample(self, features, n_samples):
        """贪心最小距离采样，确保覆盖特征空间"""
        selected = [np.random.randint(0, len(features))]
        min_dists = np.full(len(features), np.inf)

        for _ in range(n_samples - 1):
            last = features[selected[-1]]
            dists = np.linalg.norm(features - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            selected.append(int(np.argmax(min_dists)))
        return selected

    def predict(self, image):
        """返回图像级分数和像素级异常图"""
        with torch.no_grad():
            patch_feats = self._extract_patch_features(image.unsqueeze(0))  # [HW, D]

        # 计算每个 patch 到内存库的最近邻距离
        dists = torch.cdist(patch_feats, self.memory_bank)  # [HW, n_coreset]
        anomaly_scores = dists.min(dim=1).values             # [HW]

        H = W = int(anomaly_scores.shape[0] ** 0.5)
        anomaly_map = anomaly_scores.reshape(1, 1, H, W)
        # 上采样到原图大小
        anomaly_map = F.interpolate(anomaly_map, size=image.shape[-2:],
                                    mode='bilinear', align_corners=False).squeeze()
        image_score = anomaly_map.max().item()
        return image_score, anomaly_map.numpy()
```

---

## STFPM：教师-学生蒸馏训练流程

STFPM（Student-Teacher Feature Pyramid Matching）通过让学生网络模拟教师网络的正常特征，异常区域的特征差异即为异常分数：

```
训练阶段（仅用正常样本）：
  教师网络（预训练 ResNet，固定权重）→ 提取多尺度特征 {F_t^l}
  学生网络（相同结构，随机初始化）   → 提取多尺度特征 {F_s^l}
  损失 = Σ_l MSE(F_t^l, F_s^l)
  → 学生学会模拟教师在正常图像上的输出

推理阶段：
  异常分数图 = Σ_l ||F_t^l - F_s^l||^2
  → 异常区域：学生无法准确还原教师特征 → 分数高
  → 正常区域：学生与教师特征接近 → 分数低
```

```python
# STFPM 损失函数
def stfpm_loss(teacher_feats, student_feats):
    """
    teacher_feats: list of [B, C, H, W]
    student_feats: list of [B, C, H, W]
    """
    loss = 0
    for t_feat, s_feat in zip(teacher_feats, student_feats):
        # L2 归一化后计算 MSE
        t_norm = F.normalize(t_feat, dim=1)
        s_norm = F.normalize(s_feat, dim=1)
        loss += F.mse_loss(t_norm, s_norm)
    return loss
```

---

## anomalib 完整训练+推理命令

```bash
# 安装
pip install anomalib

# 训练 PatchCore（MVTec 数据集）
anomalib train \
    --model patchcore \
    --data.path ./datasets/MVTec \
    --data.category bottle \
    --model.backbone wide_resnet50_2 \
    --model.coreset_sampling_ratio 0.1

# 训练 STFPM
anomalib train \
    --model stfpm \
    --data.path ./datasets/MVTec \
    --data.category cable

# 推理 + 评估
anomalib test \
    --model patchcore \
    --data.path ./datasets/MVTec \
    --data.category bottle \
    --ckpt_path ./results/patchcore/bottle/weights/lightning/model.ckpt

# 单张图片推理
anomalib predict \
    --model patchcore \
    --data.path ./test_images/ \
    --ckpt_path ./results/patchcore/bottle/weights/lightning/model.ckpt
```

### anomalib 自定义数据集目录结构

```
my_dataset/
├── train/
│   └── good/           # 只有正常图像
│       ├── 001.jpg
│       └── ...
└── test/
    ├── good/           # 正常测试图像
    ├── crack/          # 缺陷类型1
    ├── scratch/        # 缺陷类型2
    └── ...
```

---

## 工业异常检测数据准备流程

```python
# 数据采集建议：
# 1. 拍摄环境：固定背景、固定光源（环形灯/条形灯），避免反光
# 2. 分辨率：至少 1MP，缺陷区域 ≥ 10px
# 3. 训练集：纯正常样本，建议 200-500 张
# 4. 测试集：正常 + 各类缺陷各 20-50 张

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 数据增强（仅用于训练，不影响异常特征）
train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    # 注意：不要用颜色抖动，会干扰颜色相关异常
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

---

## 真实工厂场景挑战与解决方案

| 挑战 | 表现 | 解决方案 |
|------|------|---------|
| **光照变化** | 同类零件因光照不同特征差异大 | 标准化拍摄环境；使用光照归一化预处理 |
| **零件变形** | 柔性部件位置不固定 | 增加对齐/注册步骤；使用 affine 对齐 |
| **小缺陷** | 划痕/气泡 < 5px | 使用高分辨率输入；Patch 级细粒度检测 |
| **新类缺陷** | 训练时未见过的缺陷类型 | 无监督方法天然支持；不需要缺陷标注 |
| **多类混线** | 同产线多种零件 | 分类别建立独立模型或使用多类 anomalib |

**踩坑记录**：
- MVTec 上效果好不代表在真实场景效果好，实际部署前必须采集真实数据测试
- PatchCore 推理速度与内存库大小成正比，上线前务必设置合理的 coreset_ratio（推荐 0.01~0.05）
- 像素级 AUROC 高不等于实用，关注 PRO（Per-Region Overlap）更能反映实际缺陷定位效果

## 推荐框架

- **anomalib**（Intel 出品）：统一实现 PatchCore/STFPM/FastFlow 等
  ```bash
  pip install anomalib
  ```
