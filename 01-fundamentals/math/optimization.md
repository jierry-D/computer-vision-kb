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

## 参考资料

- 《Numerical Optimization》Nocedal & Wright
- CMU 10-725 Convex Optimization
