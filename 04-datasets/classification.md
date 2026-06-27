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
