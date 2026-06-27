# 经典论文清单

> 机器视觉必读经典，建议按年份顺序阅读。

---

## 推荐阅读顺序（按影响力排序）

对于初学者，建议按以下顺序精读，循序渐进建立完整知识体系：

1. AlexNet → ResNet（分类基础）
2. Faster R-CNN → FPN → YOLO v1（检测入门）
3. FCN → U-Net → Mask R-CNN（分割基础）
4. Attention is All You Need → ViT → Swin Transformer（Transformer 时代）
5. DETR（端到端检测范式转变）
6. DenseNet / EfficientNet / MobileNet（效率优化方向）

---

## 图像分类基础

| 论文 | 年份 | 核心贡献 | 优先级 | 论文索引 |
|------|------|----------|--------|----------|
| AlexNet（ImageNet Classification with Deep CNNs） | 2012 | 深度学习 CV 起点，首次在 ImageNet 大幅超越传统方法，引入 ReLU/Dropout/数据增强 | ★★★ 必读 | NIPS 2012，作者 Krizhevsky et al. |
| VGGNet | 2014 | 统一 3×3 卷积堆叠，证明深度是性能关键，结构简洁至今仍用作特征提取器 | ★★★ 必读 | ICLR 2015，作者 Simonyan & Zisserman，arxiv: 1409.1556 |
| GoogLeNet/Inception | 2014 | Inception 模块多尺度并行卷积，大幅减少参数量，引入辅助分类器 | ★★ 推荐 | CVPR 2015，作者 Szegedy et al. |
| ResNet（Deep Residual Learning） | 2015 | 残差连接解决退化问题，使超深网络成为可能，CVPR 2016 Best Paper | ★★★ 必读 | CVPR 2016，作者 He et al.，arxiv: 1512.03385 |
| DenseNet | 2017 | 密集连接，每层与所有前层直接相连，特征复用效率极高 | ★★ 推荐 | CVPR 2017，作者 Huang et al.，arxiv: 1608.06993 |
| MobileNetV1/V2/V3 | 2017-2019 | 深度可分离卷积，极大减少计算量，移动端部署首选 | ★★ 推荐 | arxiv: 1704.04861 / 1801.04381 / 1905.02244 |
| EfficientNet | 2019 | NAS 复合缩放（宽度/深度/分辨率），用更少参数达到更高精度 | ★★ 推荐 | ICML 2019，作者 Tan & Le，arxiv: 1905.11946 |

### 论文引用关系（分类部分）

```
AlexNet (2012)
  └── VGGNet (2014)：验证深度重要性
  └── GoogLeNet (2014)：探索多尺度结构
      └── Inception-v3/v4：进一步改进 Inception
  └── ResNet (2015)：解决深度退化，被引最多
      └── DenseNet (2017)：密化残差连接
      └── MobileNet (2017)：轻量化衍生
      └── EfficientNet (2019)：NAS + 复合缩放
```

---

## 目标检测基础

| 论文 | 年份 | 核心贡献 | 优先级 | 论文索引 |
|------|------|----------|--------|----------|
| R-CNN | 2014 | 首次将 CNN 引入检测，"选择性搜索+CNN 分类"两阶段范式开创者 | ★★★ 必读 | CVPR 2014，作者 Girshick et al. |
| Fast R-CNN | 2015 | ROI Pooling 共享卷积特征，速度大幅提升，统一训练 | ★★★ 必读 | ICCV 2015，作者 Girshick，arxiv: 1504.08083 |
| Faster R-CNN | 2015 | 提出 RPN（区域提议网络），实现真正端到端，统治两阶段检测 | ★★★ 必读 | NeurIPS 2015，作者 Ren et al.，arxiv: 1506.01497 |
| YOLO v1 | 2016 | 将检测视为回归问题，单次前向传播完成检测，实时性飞跃 | ★★★ 必读 | CVPR 2016，作者 Redmon et al.，arxiv: 1506.02640 |
| SSD | 2016 | 多尺度单阶段检测，不同层预测不同尺度目标，兼顾速度与精度 | ★★ 推荐 | ECCV 2016，作者 Liu et al.，arxiv: 1512.02325 |
| Feature Pyramid Network（FPN） | 2017 | 多尺度特征融合，自顶向下路径 + 横向连接，解决小目标检测难题 | ★★★ 必读 | CVPR 2017，作者 Lin et al.，arxiv: 1612.03144 |
| RetinaNet + Focal Loss | 2017 | 提出 Focal Loss 解决类别不平衡，单阶段检测精度首次超越两阶段 | ★★★ 必读 | ICCV 2017，作者 Lin et al.，arxiv: 1708.02002 |

### 论文引用关系（检测部分）

```
R-CNN (2014)
  └── Fast R-CNN (2015)：ROI Pooling 加速
      └── Faster R-CNN (2015)：RPN 端到端
          └── FPN (2017)：多尺度特征
              └── Mask R-CNN (2017)：扩展至实例分割
              └── RetinaNet (2017)：单阶段 + Focal Loss

YOLO v1 (2016)：并行的单阶段路线
  └── YOLO v2/v3/v4/v5/v8：持续迭代
  └── SSD (2016)：类似单阶段思路
```

---

## 分割基础

| 论文 | 年份 | 核心贡献 | 优先级 | 论文索引 |
|------|------|----------|--------|----------|
| FCN（Fully Convolutional Networks） | 2015 | 第一个端到端像素级分割，将全连接层替换为卷积，引入上采样 | ★★★ 必读 | CVPR 2015，作者 Long et al. |
| U-Net | 2015 | 编码器-解码器 + 跳跃连接，医学图像分割标配，数据效率极高 | ★★★ 必读 | MICCAI 2015，作者 Ronneberger et al.，arxiv: 1505.04597 |
| Mask R-CNN | 2017 | 在 Faster R-CNN 上加 Mask 分支，统一检测+实例分割，ICCV 2017 Best Paper | ★★★ 必读 | ICCV 2017，作者 He et al.，arxiv: 1703.06870 |
| DeepLabV3+ | 2018 | ASPP（空洞空间金字塔池化）+ 编解码结构，语义分割长期 SOTA | ★★ 推荐 | ECCV 2018，作者 Chen et al.，arxiv: 1802.02611 |
| Panoptic Segmentation | 2019 | 提出全景分割任务统一实例分割和语义分割 | ★ 了解 | CVPR 2019，作者 Kirillov et al. |

### 论文引用关系（分割部分）

```
FCN (2015)：奠基
  └── U-Net (2015)：加入跳跃连接，医学图像
  └── SegNet：编解码对称结构
  └── DeepLab 系列：空洞卷积 + CRF

Faster R-CNN
  └── Mask R-CNN (2017)：加 Mask 分支
      └── 实例分割后续工作（SOLO, CondInst 等）
```

---

## Transformer 时代

| 论文 | 年份 | 核心贡献 | 优先级 | 论文索引 |
|------|------|----------|--------|----------|
| Attention is All You Need | 2017 | Transformer 原论文，自注意力机制，NLP 革命，CV 借鉴基础 | ★★★ 必读 | NeurIPS 2017，作者 Vaswani et al.，arxiv: 1706.03762 |
| DETR | 2020 | 将 Transformer 引入检测，无 NMS，端到端，开创新范式 | ★★★ 必读 | ECCV 2020，作者 Carion et al.，arxiv: 2005.12872 |
| ViT（Vision Transformer） | 2021 | 将图像切 patch 后用纯 Transformer 分类，大数据下超越 CNN | ★★★ 必读 | ICLR 2021，作者 Dosovitskiy et al.，arxiv: 2010.11929 |
| Swin Transformer | 2021 | 分层视觉 Transformer，局部窗口注意力，通用视觉骨干，ICCV 2021 Best Paper | ★★★ 必读 | ICCV 2021，作者 Liu et al.，arxiv: 2103.14030 |
| MAE（Masked Autoencoders） | 2022 | 遮盖 75% patch 做自监督预训练，高效且效果超越有监督 | ★★ 推荐 | CVPR 2022，作者 He et al.，arxiv: 2111.06377 |

### 论文引用关系（Transformer 部分）

```
Attention is All You Need (2017)
  └── BERT/GPT（NLP）→ 启发视觉
  └── ViT (2021)：直接用 Transformer 做图像分类
      └── DeiT：数据效率改进
      └── Swin Transformer (2021)：层级化，通用骨干
          └── Swin V2：更大规模
  └── DETR (2020)：Transformer 用于检测
      └── DAB-DETR / DN-DETR / DINO-DETR：改进收敛
  └── MAE (2022)：自监督预训练
```

---

## 💡 阅读建议

- **入门阶段**：先读 AlexNet、ResNet、Faster R-CNN、YOLO、ViT，掌握各任务范式
- **进阶阶段**：FPN、Focal Loss、Swin Transformer、DETR，理解现代框架设计
- **研究阶段**：MAE、DINOv2 等自监督方向，以及你所在细分方向的最新 SOTA
- **阅读配合**：每篇论文建议配合官方代码一起看，光看论文容易遗漏关键实现细节
- **引用追踪**：用 Semantic Scholar 或 Connected Papers 查看一篇论文被哪些后续工作引用，快速定位领域进展

⚠️ **注意**：不建议追求"读完所有论文"，选对高影响力的精读远比广泛浅读有效。
