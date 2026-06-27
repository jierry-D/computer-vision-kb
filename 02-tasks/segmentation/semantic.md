# 语义分割

> 对图像每个像素进行类别预测，不区分同类不同实例。

## 经典架构

### 编码器-解码器

| 模型 | 特点 |
|------|------|
| FCN | 第一个端到端语义分割网络 |
| U-Net | 跳跃连接，医学图像标配 |
| SegNet | 最大池化索引上采样 |
| DeepLabV3+ | ASPP 空洞卷积 + 解码器，工业常用 |
| HRNet | 高分辨率表示，多尺度并行 |

### Transformer-based

| 模型 | 特点 |
|------|------|
| SETR | ViT 骨干做分割 |
| SegFormer | 轻量 MiT 骨干，简洁解码器，强烈推荐 |
| Mask2Former | 统一框架，语义/实例/全景 |
| OneFormer | 单模型三任务 |

## 关键技术

- **空洞卷积（Dilated Conv）**：增大感受野不降低分辨率
- **ASPP**：多个膨胀率并行，捕获多尺度上下文
- **上采样**：双线性插值 vs 转置卷积

## 评估指标

- **mIoU**（mean Intersection over Union）：各类 IoU 均值
- **pixel Accuracy**：像素级准确率（不均衡场景不可靠）
