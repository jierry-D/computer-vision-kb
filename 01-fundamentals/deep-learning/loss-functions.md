# 损失函数

## 分类任务

| 损失 | 适用场景 |
|------|---------|
| Cross Entropy | 标准多分类 |
| Binary CE | 二分类 / 多标签 |
| Focal Loss | 类别不均衡（目标检测） |
| Label Smoothing CE | 防止过拟合，提升泛化 |

**Focal Loss**：`FL = -α(1-p)^γ log(p)`，降低易分样本权重，专注难样本。

## 回归任务

| 损失 | 特点 |
|------|------|
| L1 Loss (MAE) | 对异常值鲁棒 |
| L2 Loss (MSE) | 梯度平滑，异常值敏感 |
| Smooth L1 (Huber) | L1+L2 结合，Faster RCNN 默认 |
| IoU Loss | 直接优化检测框重叠度 |
| GIoU / DIoU / CIoU | 更完善的 IoU 损失，YOLO 系列常用 |

## 分割任务

| 损失 | 适用场景 |
|------|---------|
| BCE + Dice | 语义分割标配 |
| Lovász Loss | 直接优化 mIoU |
| Tversky Loss | 处理小目标分割 |
