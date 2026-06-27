# 人脸视觉

## 子任务

| 任务 | 说明 |
|------|------|
| 人脸检测 | 定位图像中的人脸框 |
| 人脸对齐 | 检测关键点，矫正为正脸 |
| 人脸识别 | 身份验证（1:1）/ 搜索（1:N） |
| 人脸属性 | 年龄/性别/表情/戴眼镜等 |
| 人脸生成 | GAN/Diffusion 生成或编辑 |

## 人脸检测

| 模型 | 特点 |
|------|------|
| RetinaFace | 精度高，关键点同步检测 |
| SCRFD | 速度快，适合边缘部署 |
| YOLOv8-face | YOLO 系列定制版 |

## 人脸识别

| 模型/方法 | 特点 |
|---------|------|
| ArcFace | Additive Angular Margin Loss，主流 |
| CosFace | Large Margin Cosine Loss |
| AdaFace | 自适应 margin，低质量图像强 |

### 常用流程

```python
# 1. 检测 + 对齐
# 2. 特征提取（512-d embedding）
# 3. 余弦相似度比较
similarity = F.cosine_similarity(feat1, feat2)
```

## 数据集

- **训练**：MS1MV2（5.8M 图，85k 人）
- **测试**：LFW、CFP-FP、IJB-C
