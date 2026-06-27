# 机器视觉学习路径

> 按阶段划分，每个阶段列出核心知识点与对应资源。建议按顺序推进，不跳跃。

---

## 阶段一：数学与编程基础（2-4 周）

打牢地基，后续所有内容都依赖这里。

### 必备数学
- [ ] 线性代数：矩阵乘法、特征分解、SVD → [笔记](./01-fundamentals/math/linear-algebra.md)
- [ ] 概率统计：贝叶斯、高斯分布、最大似然估计 → [笔记](./01-fundamentals/math/probability-statistics.md)
- [ ] 优化：梯度下降、链式法则、凸优化基础 → [笔记](./01-fundamentals/math/optimization.md)

### 编程工具
- [ ] Python 熟练（numpy / matplotlib）
- [ ] OpenCV 基本操作 → [笔记](./05-toolchain/libraries/opencv.md)
- [ ] PyTorch 基础（Tensor、autograd、Dataset/DataLoader）→ [笔记](./05-toolchain/libraries/pytorch-cv.md)

---

## 阶段二：图像处理基础（2-3 周）

理解像素级操作，是深度学习之前的核心储备。

- [ ] 色彩空间转换（RGB/HSV/灰度）→ [笔记](./01-fundamentals/image-processing/color-spaces.md)
- [ ] 空间域滤波（高斯模糊、锐化）→ [笔记](./01-fundamentals/image-processing/filtering.md)
- [ ] 形态学操作（腐蚀/膨胀/开闭运算）→ [笔记](./01-fundamentals/image-processing/morphology.md)
- [ ] 边缘检测（Canny、Sobel）→ [笔记](./01-fundamentals/image-processing/edge-detection.md)
- [ ] 传统特征提取（SIFT、ORB、HOG）→ [笔记](./01-fundamentals/image-processing/feature-extraction.md)

**实践：** 用 OpenCV 完成一个目标匹配或图像拼接 demo → [demos](./06-demos/image-processing/)

---

## 阶段三：相机几何与三维视觉（2-3 周）

做工业视觉、机器人、SLAM 必须掌握的部分。

- [ ] 相机模型与内外参 → [笔记](./01-fundamentals/camera-geometry/camera-model.md)
- [ ] 相机标定（张正友标定法）→ [笔记](./01-fundamentals/camera-geometry/calibration.md)
- [ ] 对极几何与基础矩阵 → [笔记](./01-fundamentals/camera-geometry/epipolar-geometry.md)
- [ ] 双目视觉与视差估计 → [笔记](./01-fundamentals/camera-geometry/stereo-vision.md)

---

## 阶段四：深度学习核心（3-4 周）

机器视觉的现代主干。

- [ ] CNN 基本组件（卷积/BN/Pooling/激活）→ [笔记](./01-fundamentals/deep-learning/cnn-basics.md)
- [ ] 主流骨干网络（ResNet/EfficientNet/MobileNet）→ [笔记](./01-fundamentals/deep-learning/architectures.md)
- [ ] 损失函数设计 → [笔记](./01-fundamentals/deep-learning/loss-functions.md)
- [ ] 数据增强策略 → [笔记](./01-fundamentals/deep-learning/data-augmentation.md)
- [ ] 训练技巧（学习率调度/混合精度/过拟合处理）→ [笔记](./01-fundamentals/deep-learning/training-tricks.md)
- [ ] Transformer 在视觉中的应用（ViT/Swin）→ [笔记](./01-fundamentals/deep-learning/transformers-in-cv.md)

**推荐资源：** Stanford CS231n → [courses](./08-resources/courses.md)

---

## 阶段五：核心视觉任务（按需选择，4-8 周）

选择与自己方向相关的任务深入。

### 通用方向
- [ ] **目标检测**：从 YOLO 入手 → [anchor-based](./02-tasks/detection/anchor-based.md) | [anchor-free](./02-tasks/detection/anchor-free.md)
- [ ] **语义分割**：SegFormer / DeepLabV3+ → [笔记](./02-tasks/segmentation/semantic.md)
- [ ] **实例分割**：Mask RCNN → [笔记](./02-tasks/segmentation/instance.md)

### 工业方向
- [ ] 异常检测（无监督）→ [笔记](./02-tasks/anomaly-detection/)
- [ ] OCR 流水线 → [笔记](./02-tasks/ocr/)

### 三维方向
- [ ] 点云处理（PointNet++）→ [笔记](./01-fundamentals/3d-vision/point-cloud.md)
- [ ] SLAM 基础 → [笔记](./01-fundamentals/3d-vision/slam-basics.md)
- [ ] NeRF / 3DGS → [笔记](./01-fundamentals/3d-vision/nerf-basics.md)

---

## 阶段六：工程落地（持续进行）

学完算法还要能部署上线。

- [ ] 模型导出 ONNX → [笔记](./05-toolchain/deployment/onnx.md)
- [ ] TensorRT 加速推理 → [笔记](./05-toolchain/deployment/tensorrt.md)
- [ ] OpenVINO（Intel 平台）→ [笔记](./05-toolchain/deployment/openvino.md)
- [ ] 标注工具使用（Labelme / CVAT）→ [笔记](./05-toolchain/annotation/)

---

## 阶段七：读论文（贯穿始终）

每个阶段同步阅读对应方向的经典论文。

- 论文笔记模板 → [_template.md](./03-papers/_template.md)
- 经典论文清单 → [classic](./03-papers/classic/)
- 多模态前沿（CLIP/SAM/Grounding DINO）→ [multimodal](./03-papers/multimodal/)

---

## 推荐学习顺序总览

```
数学基础 → 图像处理 → 相机几何
                ↓
           深度学习核心
                ↓
       选择任务方向深入学习
                ↓
          工程部署 + 持续读论文
```

---

## 参考资源汇总

- 书籍推荐 → [books.md](./08-resources/books.md)
- 课程推荐 → [courses.md](./08-resources/courses.md)
- 博客/频道 → [blogs-channels.md](./08-resources/blogs-channels.md)
