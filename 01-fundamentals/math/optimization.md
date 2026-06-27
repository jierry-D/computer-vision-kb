# 优化理论

## 核心概念

- **梯度下降**：SGD / Adam / AdamW
- **链式法则**：反向传播的数学基础
- **凸优化**：凸函数、KKT 条件
- **非线性最小二乘**：Gauss-Newton、Levenberg-Marquardt（用于 SLAM/标定）

## 在机器视觉中的应用

| 概念 | 应用场景 |
|------|---------|
| Adam 优化器 | 深度网络训练 |
| LM 算法 | 相机标定、Bundle Adjustment |
| 梯度裁剪 | 防止训练爆炸 |
| 学习率调度 | Warmup / Cosine Decay |

---

## 梯度下降变体详解

### 批量梯度下降（Batch GD）

每次使用**全部数据**计算梯度：

```
θ ← θ - η ∇_θ L(θ; X, y)
```

- 优点：梯度准确，收敛稳定
- 缺点：数据量大时极慢，不适合在线学习

### 随机梯度下降（SGD）

每次只用**一个样本**：

```
θ ← θ - η ∇_θ L(θ; xᵢ, yᵢ)
```

- 优点：更新快，有正则化效果（噪声帮助跳出局部极小）
- 缺点：梯度噪声大，收敛不稳定

### Mini-Batch SGD（实际最常用）

```python
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:  # batch_size=32~256
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
```

---

## 动量与自适应学习率优化器

### Momentum（动量法）

在梯度方向上累积历史动量，加速收敛并抑制震荡：

```
v_t = β v_{t-1} + (1-β) g_t          # 动量累积（β 通常=0.9）
θ_t = θ_{t-1} - η v_t
```

等价于：梯度方向不变时持续加速，方向改变时自动减速。

### RMSProp

自适应调整每个参数的学习率，除以历史梯度平方的移动平均：

```
v_t = β v_{t-1} + (1-β) g_t²
θ_t = θ_{t-1} - η / (√v_t + ε) × g_t
```

### Adam（Adaptive Moment Estimation）

结合 Momentum + RMSProp，是目前深度学习中最常用的优化器：

```
m_t = β₁ m_{t-1} + (1-β₁) g_t         # 一阶矩（梯度均值）
v_t = β₂ v_{t-1} + (1-β₂) g_t²        # 二阶矩（梯度方差）

m̂_t = m_t / (1 - β₁ᵗ)                  # 偏差修正
v̂_t = v_t / (1 - β₂ᵗ)                  # 偏差修正

θ_t = θ_{t-1} - η × m̂_t / (√v̂_t + ε)
```

默认超参：β₁=0.9, β₂=0.999, ε=1e-8

### AdamW（Adam + Weight Decay 解耦）

Adam 直接加 L2 正则会导致权重衰减效果被自适应学习率抵消，AdamW 将权重衰减解耦到参数更新步骤：

```
θ_t = θ_{t-1} - η × [m̂_t/(√v̂_t + ε) + λ θ_{t-1}]
                                           ↑
                               解耦的权重衰减（λ 通常=0.01）
```

**结论**：AdamW 通常比 Adam 在 Transformer 类模型上效果更好。

### 优化器对比

| 优化器 | 超参 | 适用场景 | 优点 | 缺点 |
|--------|------|---------|------|------|
| SGD | η, momentum | CV 微调预训练模型 | 泛化好 | 调参麻烦 |
| Adam | η, β₁, β₂ | 通用深度学习 | 收敛快 | 正则化弱 |
| AdamW | η, β₁, β₂, λ | Transformer, ViT | 正则化效果好 | 无 |
| RMSProp | η, β | RNN/LSTM | 适合非稳定梯度 | 无动量项 |

```python
import torch
import torch.optim as optim

model = ...  # 你的模型

# SGD with Momentum
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam
optimizer_adam = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

# AdamW（推荐用于 Transformer）
optimizer_adamw = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

---

## 学习率调度策略

### 1. Warmup（线性预热）

训练初期从小学习率线性增大到目标学习率，避免初始阶段梯度不稳定破坏预训练权重。

```python
def warmup_lambda(step, warmup_steps=1000, base_lr=1e-4, target_lr=1e-3):
    if step < warmup_steps:
        return target_lr * step / warmup_steps
    return target_lr
```

### 2. Cosine Annealing（余弦退火）

学习率按余弦函数从初始值衰减到最小值，训练末期缓慢下降有助于精细调整：

```
η_t = η_min + 1/2 (η_max - η_min)(1 + cos(πt/T))
```

```python
import torch.optim.lr_scheduler as sched

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 余弦退火
scheduler_cos = sched.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Warmup + Cosine（transformers 库）
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000
)
```

### 3. StepLR（阶梯衰减）

每隔固定 epoch 将学习率乘以 gamma：

```python
# 每 30 个 epoch 乘以 0.1
scheduler_step = sched.StepLR(optimizer, step_size=30, gamma=0.1)

# 在多个里程碑处衰减
scheduler_multi = sched.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

# 训练循环
for epoch in range(200):
    train(model, optimizer)
    scheduler_multi.step()
    print(f"Epoch {epoch}: lr = {optimizer.param_groups[0]['lr']:.2e}")
```

### 4. 各策略对比可视化

```python
import numpy as np
import matplotlib.pyplot as plt

T = 100
t = np.arange(T)

# Cosine
cos_lr = 1e-6 + 0.5 * (1e-3 - 1e-6) * (1 + np.cos(np.pi * t / T))

# Step（每30步乘0.1）
step_lr = np.array([1e-3 * (0.1 ** (i // 30)) for i in t])

# Warmup(10步) + Cosine
warmup = np.linspace(0, 1e-3, 10)
after  = 1e-6 + 0.5*(1e-3 - 1e-6)*(1 + np.cos(np.pi*np.arange(T-10)/(T-10)))
wc_lr  = np.concatenate([warmup, after])

plt.figure(figsize=(10, 4))
plt.plot(cos_lr, label='Cosine')
plt.plot(step_lr, label='StepLR')
plt.plot(wc_lr, label='Warmup+Cosine')
plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
plt.legend(); plt.yscale('log')
plt.tight_layout(); plt.show()
```

---

## LM 算法（Levenberg-Marquardt）

### 适用场景

非线性最小二乘问题：

```
min_θ  Σ ||f(xᵢ; θ) - yᵢ||²
```

在相机标定、Bundle Adjustment、ICP 点云配准中广泛使用。

### 算法原理

LM 算法在 **Gauss-Newton** 和**梯度下降**之间自适应切换：

```
(J^T J + λ I) Δθ = J^T r
```

- J：雅可比矩阵（残差对参数的偏导）
- r：残差向量
- λ：阻尼系数

当 λ→0：退化为 Gauss-Newton（收敛快，但可能震荡）
当 λ→∞：退化为梯度下降（收敛慢，但稳定）

### 详细步骤

1. 初始化参数 θ₀，设置 λ=0.001
2. 计算残差 r = f(θ) - y，代价 F = ||r||²
3. 计算雅可比 J（数值差分或解析推导）
4. 求解增量 Δθ：`(J^T J + λ I) Δθ = -J^T r`
5. 计算新代价 F_new = ||f(θ + Δθ) - y||²
6. 若 F_new < F：接受更新 θ ← θ + Δθ，减小 λ（λ /= 10）
7. 若 F_new ≥ F：拒绝更新，增大 λ（λ *= 10）
8. 重复步骤 2-7 直到收敛（||Δθ|| < ε 或 ||J^T r|| < ε）

```python
import numpy as np

def lm_optimize(f, J_func, theta0, max_iter=100, lam=0.001, tol=1e-8):
    """
    简化 LM 算法实现
    f: 残差函数 f(theta) -> r
    J_func: 雅可比函数 J(theta) -> J
    theta0: 初始参数
    """
    theta = theta0.copy()
    for i in range(max_iter):
        r = f(theta)
        J = J_func(theta)
        F = r @ r

        # 求解线性系统
        A = J.T @ J + lam * np.eye(len(theta))
        g = J.T @ r
        delta = np.linalg.solve(A, -g)

        theta_new = theta + delta
        F_new = f(theta_new) @ f(theta_new)

        if F_new < F:
            theta = theta_new
            lam  = max(lam / 10, 1e-16)
            if np.linalg.norm(delta) < tol:
                print(f"收敛于第 {i} 次迭代")
                break
        else:
            lam = min(lam * 10, 1e16)
    return theta
```

---

## 参考资料

- 《Numerical Optimization》Nocedal & Wright
- CMU 10-725 Convex Optimization
-《视觉 SLAM 十四讲》高翔（LM 与 Bundle Adjustment 详解）
- PyTorch 优化器文档：https://pytorch.org/docs/stable/optim.html
