# 训练技巧

## 学习率策略

| 策略 | 说明 |
|------|------|
| Warmup | 前 N 步从小 LR 线性升至目标 LR，避免初期震荡 |
| Cosine Decay | 余弦退火，收敛更平滑 |
| Step Decay | 每隔固定 epoch 乘以衰减因子 |
| OneCycleLR | warmup + cosine decay，超收敛 |

## 正则化

- **Dropout**：随机丢弃神经元，防过拟合（Transformer 常用）
- **Weight Decay**：L2 正则，AdamW 默认解耦
- **Stochastic Depth**（Drop Path）：随机丢弃整个残差块，Swin/DeiT 常用
- **EMA**（指数移动平均）：维护权重的平滑版本，提升推理精度

## 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = model(x)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 梯度相关

- **梯度裁剪**：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **梯度累积**：小显存模拟大 batch

## 初始化

- Kaiming（He）初始化：ReLU 网络标准初始化
- Xavier 初始化：Sigmoid/Tanh 网络
