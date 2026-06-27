# 数据增强

## 基础增强

| 方法 | 效果 |
|------|------|
| 随机翻转 | 方向不变性 |
| 随机裁剪/缩放 | 尺度、位置不变性 |
| 颜色抖动（亮度/对比度/饱和度） | 光照不变性 |
| 随机旋转 | 方向泛化 |
| 高斯噪声/模糊 | 鲁棒性 |

## 高级增强

| 方法 | 说明 | 适用 |
|------|------|------|
| Mixup | 两张图线性叠加 | 分类 |
| CutMix | 裁剪区域替换 | 分类/检测 |
| Mosaic | 4 张图拼合 | YOLO 检测 |
| Copy-Paste | 实例粘贴到新背景 | 实例分割 |
| AutoAugment / RandAugment | NAS 搜索最优策略 | 分类 |
| AugMix | 多路增强混合 | 鲁棒性 |

## 推荐库

- **Albumentations**：速度快，支持检测/分割的 bbox/mask 同步增强
- **torchvision.transforms v2**：PyTorch 官方，支持 bbox 同步

```python
import albumentations as A
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Mosaic(p=0.5),
], bbox_params=A.BboxParams(format='yolo'))
```
