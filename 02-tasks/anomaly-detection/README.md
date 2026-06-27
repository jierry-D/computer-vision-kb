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

## 推荐框架

- **anomalib**（Intel 出品）：统一实现 PatchCore/STFPM/FastFlow 等
  ```bash
  pip install anomalib
  ```
