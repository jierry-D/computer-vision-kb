# 推荐课程

## 必修课

| 课程 | 机构 | 特点 |
|------|------|------|
| CS231n: CNN for Visual Recognition | Stanford | CV 必修，全英，有中文字幕 |
| 动手学深度学习（d2l.ai 配套课） | 李沐 | 中文，实战，B 站有录播 |

## 深度学习基础

| 课程 | 机构 | 特点 |
|------|------|------|
| Deep Learning Specialization | deeplearning.ai | 吴恩达，系统入门 |
| 11-785 Deep Learning | CMU | 理论扎实 |
| Fast.ai Practical Deep Learning | fast.ai | 自顶向下，快速上手 |

## 三维视觉 / SLAM

| 课程 | 特点 |
|------|------|
| 《视觉 SLAM 十四讲》配套代码 | 高翔，中文 SLAM 入门标配 |
| ETH Zurich: 3D Vision | 相机几何、三维重建系统 |

## 工业实战

| 课程 | 特点 |
|------|------|
| Roboflow Universe 教程系列 | 检测数据标注到部署全流程 |
| Ultralytics 官方文档 | YOLO 最佳实践 |

## B 站推荐 UP 主

- **跟李沐学 AI**：论文精读，强烈推荐
- **DR_CAN**：控制+状态估计，SLAM 相关
- **霹雳吧啦 Wz**：PyTorch 实战，检测分类

---

## 各课程详细章节推荐

### CS231n: CNN for Visual Recognition（Stanford）

**视频来源**：
- YouTube：搜索 "Stanford CS231n 2017"（经典版）或 "CS231n 2022"（最新版）
- B 站：搜索 "CS231n"，有带中文字幕的版本

**重点讲次**：

| 讲次 | 主题 | 重要性 |
|------|------|--------|
| Lecture 1-2 | 图像分类与 kNN/线性分类器 | ★★★ 基础必看 |
| Lecture 3 | 损失函数与优化 | ★★★ 必看 |
| Lecture 4 | 反向传播与神经网络 | ★★★ 必看 |
| Lecture 5 | 卷积神经网络 | ★★★ 必看 |
| Lecture 6 | 训练技巧（BN/Dropout/初始化）| ★★★ 必看 |
| Lecture 7 | CNN 架构（AlexNet/VGG/GoogLeNet/ResNet）| ★★★ 必看 |
| Lecture 8 | 深度学习软件（PyTorch/TF）| ★★ 推荐 |
| **Lecture 11** | **目标检测与分割** | ★★★ CV 核心，必看 |
| **Lecture 12** | **可视化与理解** | ★★★ 理解模型行为 |
| Lecture 13 | 生成模型（VAE/GAN）| ★★ 推荐 |
| Lecture 15 | 强化学习基础 | ★ 可选 |

**作业难度与收获**：
- A1（kNN/SVM/Softmax）：★★ 基础，建议完成
- A2（BatchNorm/Dropout/CNN）：★★★ 核心，必须做
- A3（RNN/LSTM/Transformer）：★★★ 进阶，NLP 背景者做
- 完成所有作业约需 40-60 小时

💡 **Tips**：可以只看 Lecture 1-8 和 Lecture 11-12，跳过 NLP/RL 相关讲次，专注 CV 内容，节省时间。

---

### 动手学深度学习（李沐，B 站课程）

**视频来源**：
- B 站：搜索 "跟李沐学AI" 或 "动手学深度学习"
- YouTube：搜索 "李沐 d2l"
- 课程主页：courses.d2l.ai

**重点视频推荐**：

| 视频系列 | 内容 | 重要性 |
|---------|------|--------|
| 线性神经网络系列（3讲）| 最小二乘/Softmax 回归 | ★★★ |
| 多层感知机系列（3讲）| MLP/过拟合/丢弃法 | ★★★ |
| 深度学习计算（2讲）| 参数管理/自定义层 | ★★ |
| CNN 经典系列（6讲）| AlexNet/VGG/ResNet 等 | ★★★ |
| **目标检测系列（8讲）** | 锚框/R-CNN/YOLO/SSD | ★★★ 重点 |
| **语义分割（2讲）** | FCN/转置卷积 | ★★★ 重点 |
| 注意力机制系列（4讲）| Transformer/自注意力 | ★★★ |
| 论文精读系列 | ResNet/BERT/ViT/MAE 等 | ★★★ 强烈推荐 |

**作业收获评价**：d2l 的 Jupyter notebook 配套质量极高，边读代码边运行，是目前最好的中文 CV 学习资源之一。

---

### Deep Learning Specialization（吴恩达，Coursera）

**视频来源**：
- Coursera（付费完整版，证书）
- YouTube 有部分免费试看
- B 站有搬运版（搜索"吴恩达深度学习专项课程"）

**5 门子课程建议**：

| 课程 | 内容 | 建议 |
|------|------|------|
| 1. 神经网络与深度学习 | 基础，必学 | 全部看 |
| 2. 改善深层神经网络 | 优化技巧，必学 | 全部看 |
| 3. 结构化机器学习项目 | 工程实践，实用 | 全部看 |
| 4. 卷积神经网络 | CV 核心，必学 | 全部看 |
| 5. 序列模型 | RNN/Transformer，可选 | CV 方向可跳过前 3 周 |

**作业难度**：★★☆☆☆，以 Python 代码填空为主，适合验证理解。

---

### Fast.ai Practical Deep Learning（fast.ai）

**视频来源**：
- 官网：course.fast.ai（全部免费，含视频+Jupyter notebook）
- YouTube：搜索 "fast.ai 2022"

**课程特点**：采用"自顶向下"教学法，第 1 课就能跑出 SOTA 模型，逐渐深入原理。

**推荐学习路线**：
- Part 1（7讲）：直接用 fastai 库完成分类、检测、分割、NLP 任务
- Part 2（8讲）：从头实现深度学习基础（更深，适合进阶）

**适合人群**：有编程基础、想快速看到效果的实践者；不适合想从数学推导起步的人。

---

## 快速补充路线（2 周 CV 速成建议）

适合已有 Python 基础，需要快速了解 CV 的工程师：

```
第 1-3 天：深度学习基础
  - 《动手学深度学习》第 4-6 章（MLP/训练技巧）
  - 跑通 PyTorch 基础代码（MNIST 分类）

第 4-6 天：CV 核心（图像分类）
  - 《动手学深度学习》第 7-8 章（CNN/ResNet）
  - CS231n Lecture 5-7 视频
  - 动手训练 CIFAR-10

第 7-9 天：目标检测入门
  - 《动手学深度学习》目标检测系列视频（4讲）
  - 跑通 Ultralytics YOLOv8（5 行代码检测）
  - 在自己的数据集上微调 YOLO

第 10-12 天：分割与部署
  - 跑通 MMSegmentation 或 SAM
  - ONNX 导出和推理验证
  - 了解 TensorRT 基础概念

第 13-14 天：巩固与实战
  - 选一个 Kaggle CV 比赛（分类赛道入门）
  - 完成 CS231n A2 作业（可选）
```

⚠️ **注意**：2 周速成只能建立基本框架，无法深入理解原理。建议之后持续精读 3-5 篇经典论文，并在真实项目中实践，才能形成稳固的知识体系。
