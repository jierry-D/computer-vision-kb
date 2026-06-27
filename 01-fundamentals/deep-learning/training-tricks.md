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

---

## 完整 PyTorch 训练循环（含混合精度/梯度累积/EMA）

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from copy import deepcopy

class EMA:
    """指数移动平均：维护模型参数的平滑版本，用于推理"""
    def __init__(self, model, decay=0.9999):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.copy_(ema_p * self.decay + model_p.data * (1.0 - self.decay))


def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    ema=None, accumulate_steps=4, clip_grad=1.0,
                    device='cuda'):
    """
    完整训练循环：混合精度 + 梯度累积 + EMA + 梯度裁剪
    accumulate_steps: 梯度累积步数（模拟 batch_size * accumulate_steps 的大 batch）
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 混合精度前向
        with autocast():
            logits = model(images)
            loss = nn.CrossEntropyLoss()(logits, labels)
            # 梯度累积：缩放 loss
            loss = loss / accumulate_steps

        # 反向传播（scaler 防止 fp16 下溢）
        scaler.scale(loss).backward()

        if (step + 1) % accumulate_steps == 0:
            # 梯度裁剪（先 unscale 再裁剪）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # EMA 更新
            if ema is not None:
                ema.update(model)

        total_loss += loss.item() * accumulate_steps

    # 学习率调度
    scheduler.step()
    return total_loss / len(loader)


# 完整初始化示例
model = MyModel().to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scaler = GradScaler()
ema = EMA(model, decay=0.9999)

from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

---

## 优化器配置

### AdamW（推荐）

AdamW 将 weight decay 从梯度更新中解耦，避免 Adam 中 L2 正则化与自适应学习率的相互干扰。

```python
# 常见配置：对 bias 和 LayerNorm 参数不应用 weight decay
def get_param_groups(model, weight_decay=0.05):
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith('.bias'):
            # bias、LayerNorm 的 weight/bias 不加 weight decay
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {'params': decay_params,    'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

optimizer = torch.optim.AdamW(
    get_param_groups(model, weight_decay=0.05),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

### 各优化器对比

| 优化器 | 特点 | 适用场景 |
|--------|------|----------|
| SGD + Momentum | 需仔细调 LR，收敛稳定 | ResNet/YOLO 等 CNN |
| Adam | 自适应 LR，对 LR 不敏感 | NLP、小数据集 |
| AdamW | Adam + 解耦 weight decay | Transformer、ViT、通用首选 |
| LAMB | 大 batch 训练专用 | 超大 batch（>8192）|
| Lion | 内存占用更低，速度更快 | 大模型训练 |

---

## Cosine Annealing with Warmup 实现

Warmup 阶段线性增大学习率，避免训练初期因大梯度引起的震荡；Cosine Decay 阶段平滑降低学习率。

```python
import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps,
                                     min_lr_ratio=0.0):
    """
    warmup_steps: 预热步数（通常为总步数的 5%~10%）
    total_steps: 总训练步数
    min_lr_ratio: 最低 LR 与初始 LR 的比值（默认 0，即衰减到 0）
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 线性 Warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine Decay
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


# 使用示例
num_epochs = 100
steps_per_epoch = len(train_loader)
total_steps = num_epochs * steps_per_epoch
warmup_steps = int(0.05 * total_steps)  # 5% 的步数做 warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer, warmup_steps=warmup_steps, total_steps=total_steps
)

# 训练循环中每步（而非每 epoch）调用
for step, batch in enumerate(train_loader):
    # ... 前向+反向 ...
    scheduler.step()
```

---

## 过拟合诊断方法

| 症状 | 诊断方法 | 解决方案 |
|------|----------|----------|
| 训练 loss 低，验证 loss 高 | 绘制 train/val loss 曲线 | 增加数据/数据增强/正则化/减小模型 |
| 验证 loss 不降 | 检查数据质量和标注 | 修正标注，调整 LR |
| 训练 loss 震荡 | 绘制每步 loss，检查 LR | 降低 LR，使用 warmup |
| 验证 acc 前期高后期下降 | 检查是否遗忘 | Cosine LR + EMA |

**实用诊断代码**：

```python
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses,   label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()

    ax2.plot(val_accs, label='Val Acc', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
```

---

## 常见训练 Bug 排查清单

**损失异常**

- [ ] loss 为 NaN → 检查 LR 是否过大、梯度裁剪是否生效、数据中是否有 inf/NaN
- [ ] loss 不下降 → 检查 optimizer.zero_grad() 是否在正确位置、数据 pipeline 是否正确
- [ ] loss 震荡剧烈 → 降低 LR，加入 warmup，检查 batch size

**精度异常**

- [ ] 训练 acc 高验证 acc 低（过拟合）→ 增加 dropout/weight decay/数据增强
- [ ] 训练验证 acc 均低（欠拟合）→ 增大模型/调高 LR/检查数据标注质量
- [ ] 验证 acc 跑飞（异常高）→ 检查是否有数据泄漏（验证集被训练）

**显存问题**

- [ ] CUDA OOM → 减小 batch size，使用混合精度，使用梯度累积
- [ ] 显存随 epoch 增大 → 检查是否有变量没 detach（如把 tensor 存入 list）

**速度问题**

- [ ] GPU 利用率低 → 增加 DataLoader 的 num_workers，使用 pin_memory=True
- [ ] 训练比预期慢 → 检查数据预处理是否在 CPU 上成瓶颈，考虑缓存预处理结果

**其他**

- [ ] 复现性问题 → 设置所有随机种子
  ```python
  import torch, random, numpy as np
  def set_seed(seed=42):
      random.seed(seed); np.random.seed(seed)
      torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
  ```
- [ ] 模型保存/加载后精度下降 → 检查是否保存了 model.state_dict()（推荐），而非整个 model
- [ ] 多 GPU 训练结果不一致 → 检查 SyncBN 是否启用，BN 统计是否同步
