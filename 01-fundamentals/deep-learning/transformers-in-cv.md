# Transformer 在视觉中的应用

## 核心机制

- **自注意力（Self-Attention）**：每个 token 与所有其他 token 计算关联
- **位置编码**：补充位置信息（绝对/相对）
- **Multi-Head Attention**：多组注意力并行，捕捉不同子空间特征

## 主要模型

### ViT（Vision Transformer）
- 将图像切成固定 patch（16×16），展平后当 token
- 纯 Transformer 编码器
- 缺点：需要大量数据预训练

### Swin Transformer
- 分层结构（类似 CNN 的多尺度）
- 窗口注意力（W-MSA）+ 移位窗口（SW-MSA），降低计算复杂度
- 目前检测/分割最强骨干之一

### DeiT
- 数据高效的 ViT，加入蒸馏 token

### DETR
- 用 Transformer 做端到端目标检测，去掉 NMS
- 延伸：Deformable DETR、DINO（检测）

## 与 CNN 对比

| 维度 | CNN | Transformer |
|------|-----|------------|
| 归纳偏置 | 强（局部性、平移不变） | 弱（需更多数据） |
| 全局感受野 | 需堆叠多层 | 单层即可 |
| 计算复杂度 | O(n) | O(n²)，Swin 改为 O(n) |
| 小数据集 | 更好 | 较差（需预训练） |
