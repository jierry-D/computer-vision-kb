# CNN 基础

## 核心组件

### 卷积层
- 感受野、步长（stride）、填充（padding）
- 输出尺寸：`(W - K + 2P) / S + 1`
- 深度可分离卷积（Depthwise Separable Conv）：MobileNet 的核心，减少参数量

### 批归一化（Batch Normalization）
- 加速训练，缓解梯度消失
- 训练时用 batch 统计，推理时用滑动平均
- 变体：LayerNorm（Transformer）、GroupNorm（小 batch）

### 激活函数

| 函数 | 优点 | 缺点 |
|------|------|------|
| ReLU | 计算简单，无梯度消失 | Dead Neuron |
| LeakyReLU | 解决 Dead Neuron | — |
| GELU | Transformer 常用 | 计算略复杂 |
| Sigmoid/Tanh | 输出有界 | 深层梯度消失 |

### 池化层
- MaxPooling：保留最强特征
- AvgPooling：全局平均池化（GAP）常用于分类头
- AdaptiveAvgPool：输出固定尺寸，与输入无关
