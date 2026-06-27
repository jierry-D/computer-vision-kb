# 图像分类数据集

| 数据集 | 规模 | 类别 | 特点 |
|--------|------|------|------|
| MNIST | 70k | 10 | 手写数字，入门 |
| CIFAR-10/100 | 60k | 10/100 | 小图（32×32），快速实验 |
| ImageNet-1K | 1.28M | 1000 | 分类基准标准 |
| ImageNet-21K | 14M | 21841 | 大规模预训练 |
| Places365 | 1.8M | 365 | 场景分类 |
| iNaturalist | 859k | 5089 | 细粒度，生物分类 |
| CUB-200-2011 | 11.8k | 200 | 细粒度，鸟类 |

## 常用预处理

```python
from torchvision import transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  # ImageNet 统计值
])
```

---

## 数据集下载（torchvision 方式）

torchvision 内置了常见数据集的自动下载功能，推荐优先使用：

```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# MNIST
mnist_train = datasets.MNIST(
    root="./data", train=True, download=True,
    transform=transforms.ToTensor()
)

# CIFAR-10
cifar10_train = datasets.CIFAR10(
    root="./data", train=True, download=True,
    transform=transforms.ToTensor()
)

# CIFAR-100
cifar100_train = datasets.CIFAR100(
    root="./data", train=True, download=True,
    transform=transforms.ToTensor()
)

# ImageNet（需要手动下载，torchvision 只支持读取已下载的）
# 从 image-net.org 下载后：
imagenet = datasets.ImageNet(
    root="/data/imagenet",   # 包含 train/ 和 val/ 子目录
    split="train",
    transform=transform
)

# Places365（自动下载，但文件较大）
places = datasets.Places365(
    root="./data", split="train-standard",
    small=True,  # 使用小图版本（256px），完整版约 100GB
    download=True,
    transform=transform
)
```

### ImageNet 目录结构

```
imagenet/
├── train/
│   ├── n01440764/    # 每个类别一个文件夹（WordNet ID）
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

⚠️ **注意**：ImageNet 官方下载需注册账号，国内推荐从学校/机构的镜像或 Academic Torrents 下载，完整数据集约 150GB。

---

## ImageNet 预训练权重使用建议

### 直接加载预训练模型

```python
import torchvision.models as models

# 加载预训练 ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# 修改分类头适应自定义任务
import torch.nn as nn
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

### 迁移学习三种策略

| 策略 | 适用场景 | 代码 |
|------|----------|------|
| 冻结全部特征层，只训练分类头 | 数据量很少（<1000张/类）| `for p in model.parameters(): p.requires_grad = False` + 解冻最后一层 |
| 微调全部参数（小学习率）| 数据量中等，与 ImageNet 分布相近 | 全参数训练，骨干用 1e-4，头部用 1e-3 |
| 从头训练 | 数据量大（>100k），与 ImageNet 分布差异大 | 不加载预训练权重 |

```python
# 分层学习率（推荐）
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(),     "lr": 1e-3},
], lr=1e-4)
```

💡 **Tips**：ImageNet 预训练权重对自然图像效果很好，但对医学图像、工业图像等特殊域迁移效果可能不理想，此时考虑在领域数据上做自监督预训练（MAE/MoCo）再微调。

---

## 数据集划分比例建议

| 数据量级 | 训练 | 验证 | 测试 | 建议 |
|----------|------|------|------|------|
| <1000 张 | 70% | 15% | 15% | 考虑 k-fold 交叉验证 |
| 1k~10k 张 | 80% | 10% | 10% | 标准划分 |
| >10k 张 | 85% | 7.5% | 7.5% | 验证集不需要太多 |
| 工业场景 | 尽量多 | 10% | 独立测试集 | 测试集必须覆盖所有缺陷类型 |

⚠️ **注意**：细粒度分类和类别不均衡场景中，划分时要确保每个类别在各集合中的比例一致（分层采样，使用 `sklearn.model_selection.train_test_split` 的 `stratify` 参数）。

```python
from sklearn.model_selection import train_test_split

# 分层划分
X_train, X_test, y_train, y_test = train_test_split(
    file_paths, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels    # 关键：按类别比例划分
)
```

---

## 类别不均衡处理方法

工业检测、医疗诊断等场景中类别极度不均衡（正常:缺陷 = 100:1），需要特殊处理。

### 方法一：类别权重（简单有效）

```python
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# 计算每个样本的采样权重
class_counts = [1000, 50, 30]   # 各类别样本数
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

# 为每个样本分配权重
sample_weights = [class_weights[label] for label in all_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))

dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### 方法二：损失函数权重

```python
# CrossEntropyLoss 加权
weights = torch.tensor([1.0, 20.0, 33.0])  # 少数类权重更高
criterion = nn.CrossEntropyLoss(weight=weights.cuda())
```

### 方法三：过采样（Oversampling）

```python
# 使用 imbalanced-learn 库
from imblearn.over_sampling import RandomOverSampler, SMOTE

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# SMOTE（合成少数类样本，对图像数据不直接适用，需先提取特征）
```

### 方法四：Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

---

## 数据集可视化代码

```python
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def show_dataset_samples(dataset, n_rows=3, n_cols=6, figsize=(15, 8)):
    """展示数据集样例图片"""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # 获取类别名称
    class_names = dataset.classes if hasattr(dataset, 'classes') else None

    for i, ax in enumerate(axes.flat):
        if i >= len(dataset):
            ax.axis('off')
            continue

        img, label = dataset[i]

        # 反归一化（如果做过归一化）
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)

        ax.imshow(img)
        title = class_names[label] if class_names else str(label)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=150)
    plt.show()

# 使用示例
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
show_dataset_samples(dataset)
```

```python
# 类别分布可视化
from collections import Counter

labels = [dataset[i][1] for i in range(len(dataset))]
label_counts = Counter(labels)
class_names = dataset.classes

plt.figure(figsize=(12, 5))
plt.bar(class_names, [label_counts[i] for i in range(len(class_names))])
plt.xticks(rotation=45, ha='right')
plt.title("类别分布")
plt.ylabel("样本数量")
plt.tight_layout()
plt.show()
```
