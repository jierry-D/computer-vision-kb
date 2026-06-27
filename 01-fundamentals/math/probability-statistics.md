# 概率与统计

## 核心概念

- **概率分布**：高斯分布、伯努利分布、多项分布
- **贝叶斯定理**：先验/后验/似然
- **最大似然估计（MLE）**：损失函数推导基础
- **信息论**：熵、KL 散度、交叉熵

## 在机器视觉中的应用

| 概念 | 应用场景 |
|------|---------|
| 高斯分布 | 背景建模（GMM）、卡尔曼滤波 |
| 贝叶斯 | 贝叶斯分类器、概率目标检测 |
| 交叉熵 | 分类任务损失函数 |
| KL 散度 | VAE、分布对齐 |

---

## 贝叶斯公式详解

### 公式

```
P(A|B) = P(B|A) × P(A) / P(B)
```

- `P(A)`：**先验概率**，观测前对 A 的置信度
- `P(B|A)`：**似然**，已知 A 时观测到 B 的概率
- `P(B)`：**证据**，归一化常数
- `P(A|B)`：**后验概率**，观测到 B 后对 A 的更新置信度

### 贝叶斯推断示例：图像分类

```python
import numpy as np

# 先验：某类别在数据集中的频率
prior = {"cat": 0.3, "dog": 0.5, "bird": 0.2}

# 似然：给定类别，观测到特征（如耳朵形状）的概率
likelihood_pointy_ear = {"cat": 0.8, "dog": 0.2, "bird": 0.05}

# 计算后验（分子）
posterior_unnorm = {
    cls: prior[cls] * likelihood_pointy_ear[cls]
    for cls in prior
}

# 归一化（除以证据 P(B)）
evidence = sum(posterior_unnorm.values())
posterior = {cls: v / evidence for cls, v in posterior_unnorm.items()}
print(posterior)
# → cat 的后验概率大幅上升
```

### 全概率公式（贝叶斯分母展开）

```
P(B) = Σ P(B|Aᵢ) × P(Aᵢ)
```

---

## 高斯（正态）分布

### 一维高斯分布

概率密度函数：

```
f(x) = 1/(σ√(2π)) × exp(-(x-μ)²/(2σ²))
```

- μ：均值（分布中心）
- σ²：方差（分布宽度）

### 多维高斯分布

```
f(x) = 1/((2π)^(d/2)|Σ|^(1/2)) × exp(-1/2 (x-μ)^T Σ^{-1} (x-μ))
```

- μ：d 维均值向量
- Σ：d×d 协方差矩阵（半正定）

### 重要性质

1. **线性变换封闭性**：若 X~N(μ,Σ)，则 AX+b~N(Aμ+b, AΣA^T)
2. **独立高斯之和仍是高斯**：N(μ₁,σ₁²) + N(μ₂,σ₂²) = N(μ₁+μ₂, σ₁²+σ₂²)
3. **68-95-99.7 规则**：μ±σ 包含68%，μ±2σ 包含95%，μ±3σ 包含99.7%的概率

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

# 一维高斯分布
mu, sigma = 0, 1
x = np.linspace(-4, 4, 200)
pdf = norm.pdf(x, mu, sigma)
plt.plot(x, pdf)
plt.fill_between(x, pdf, where=(np.abs(x) < 1), alpha=0.3, label='μ±σ (68%)')
plt.legend()

# 二维高斯分布可视化
mean = np.array([0, 0])
cov  = np.array([[1, 0.7], [0.7, 1]])
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
pos  = np.dstack((xx, yy))
rv   = multivariate_normal(mean, cov)
plt.contourf(xx, yy, rv.pdf(pos), levels=20)
plt.colorbar()
plt.show()
```

---

## EM 算法（期望最大化）

EM 算法用于含隐变量的最大似然估计，典型应用是高斯混合模型（GMM）。

### 算法框架

**E 步（Expectation）**：固定参数 θ，计算隐变量的后验分布（软分配）：
```
Q(θ|θ_old) = E_{z|x,θ_old}[log P(x,z|θ)]
```

**M 步（Maximization）**：最大化 Q 函数，更新参数：
```
θ_new = argmax Q(θ|θ_old)
```

### GMM 的 EM 推导

```python
import numpy as np
from sklearn.mixture import GaussianMixture

# 生成双峰数据
np.random.seed(42)
data = np.concatenate([
    np.random.randn(300, 2) + [2, 2],
    np.random.randn(200, 2) + [-2, -2]
])

# 拟合 GMM（内部使用 EM）
gmm = GaussianMixture(n_components=2, covariance_type='full', max_iter=100)
gmm.fit(data)

print("均值:\n", gmm.means_)
print("协方差:\n", gmm.covariances_)
print("混合权重:", gmm.weights_)

# 预测软分配
probs = gmm.predict_proba(data)  # shape: (N, K)
labels = gmm.predict(data)       # 硬分配
```

---

## 卡尔曼滤波

### 适用场景

连续动态系统的状态估计，假设：
1. 状态转移为线性
2. 噪声为高斯分布

典型应用：目标跟踪、传感器融合、SLAM

### 核心方程

**系统模型**：
```
x_k = F x_{k-1} + B u_k + w_k    (状态转移，w_k~N(0,Q))
z_k = H x_k + v_k                 (观测模型，v_k~N(0,R))
```

**预测步骤**：
```
x̂_{k|k-1} = F x̂_{k-1|k-1} + B u_k      (先验状态估计)
P_{k|k-1}  = F P_{k-1|k-1} F^T + Q        (先验协方差)
```

**更新步骤**：
```
K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}   (卡尔曼增益)
x̂_{k|k} = x̂_{k|k-1} + K_k(z_k - H x̂_{k|k-1})    (后验估计)
P_{k|k}  = (I - K_k H) P_{k|k-1}                   (后验协方差)
```

```python
import numpy as np

class KalmanFilter1D:
    """一维匀速运动卡尔曼滤波"""
    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=1.0):
        self.dt = dt
        # 状态：[位置, 速度]
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1000  # 初始协方差（大 = 不确定）

        # 状态转移矩阵（匀速运动）
        self.F = np.array([[1, dt], [0, 1]])
        # 观测矩阵（只观测位置）
        self.H = np.array([[1, 0]])
        # 过程噪声协方差
        self.Q = np.eye(2) * process_noise
        # 测量噪声协方差
        self.R = np.array([[measurement_noise]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0]

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        y = z - self.H @ self.x                    # 新息
        self.x = self.x + K * y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0, 0]

# 使用示例
kf = KalmanFilter1D(dt=1.0, process_noise=0.1, measurement_noise=5.0)
measurements = [1.0, 2.1, 3.0, 3.9, 5.2]
for z in measurements:
    pred = kf.predict()
    est  = kf.update(z)
    print(f"测量: {z:.1f}, 预测: {pred:.2f}, 估计: {est:.2f}")
```

---

## 信息熵与信息论

### 香农熵（Shannon Entropy）

衡量随机变量的**不确定性**或**信息量**：

```
H(X) = -Σ P(xᵢ) log₂ P(xᵢ)
```

**直觉**：
- 若结果完全确定（P=1），H=0，不需要任何信息
- 若所有结果等可能（均匀分布），H 最大

```python
import numpy as np

def entropy(probs):
    """计算离散分布的香农熵（bits）"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # 过滤掉 0（0 log 0 = 0）
    return -np.sum(probs * np.log2(probs))

print(entropy([1.0]))              # 0.0 bits（确定）
print(entropy([0.5, 0.5]))         # 1.0 bit（二元不确定）
print(entropy([0.25]*4))           # 2.0 bits（四选一）
```

### 交叉熵损失（Cross-Entropy Loss）

真实分布 p，预测分布 q 之间的交叉熵：

```
H(p, q) = -Σ p(xᵢ) log q(xᵢ)
```

在分类任务中，`p` 为 one-hot 标签，`q` 为 softmax 输出：

```python
import numpy as np

def cross_entropy_loss(logits, labels):
    """
    logits: shape (N, C)，未经 softmax 的原始输出
    labels: shape (N,)，真实类别索引
    """
    N = logits.shape[0]
    # softmax
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    # NLL loss
    log_probs = np.log(probs[np.arange(N), labels] + 1e-9)
    return -log_probs.mean()
```

### KL 散度

衡量分布 q 相对于真实分布 p 的"额外信息量"：

```
KL(p||q) = Σ p(xᵢ) log(p(xᵢ)/q(xᵢ))
```

**注意**：KL 散度不对称，`KL(p||q) ≠ KL(q||p)`。

在 VAE 中，KL 散度用于约束隐空间接近标准高斯分布：

```
L_KL = -1/2 Σ(1 + log σ² - μ² - σ²)
```

---

## 参考资料

- 《Pattern Recognition and Machine Learning》Bishop
- 《Probabilistic Robotics》Thrun, Burgard, Fox（卡尔曼滤波）
- 《Deep Learning》Goodfellow（信息论章节）
- 《The Elements of Statistical Learning》Hastie et al.
