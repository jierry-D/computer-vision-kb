# 主流骨干网络

## 经典架构演进

| 年份 | 网络 | 核心贡献 |
|------|------|---------|
| 2012 | AlexNet | 深度学习 CV 起点，GPU 训练 |
| 2014 | VGG | 统一 3×3 卷积堆叠 |
| 2014 | GoogLeNet/Inception | 多尺度并行卷积（Inception 模块） |
| 2015 | ResNet | 残差连接，解决退化问题 |
| 2017 | DenseNet | 密集连接，特征复用 |
| 2017 | MobileNetV1 | 深度可分离卷积，轻量化 |
| 2018 | EfficientNet | NAS 复合缩放 |
| 2020 | RegNet | 可规律化设计空间 |
| 2021 | ViT | 纯 Transformer 做分类 |
| 2021 | Swin Transformer | 分层 + 窗口注意力，通用视觉骨干 |
| 2022 | ConvNeXt | 现代化 CNN，对标 Swin |

## 选型建议

| 场景 | 推荐 |
|------|------|
| 移动端部署 | MobileNetV3 / EfficientNet-Lite |
| 高精度桌面端 | Swin-L / ConvNeXt-L |
| 检测/分割骨干 | ResNet50 / Swin-T / ConvNeXt-T |
| 快速实验 | ResNet50 |
