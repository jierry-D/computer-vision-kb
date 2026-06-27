# 损失函数

## 分类任务

| 损失 | 适用场景 |
|------|---------|
| Cross Entropy | 标准多分类 |
| Binary CE | 二分类 / 多标签 |
| Focal Loss | 类别不均衡（目标检测）|
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

---

## PyTorch 代码实现

### 分类损失

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 标准交叉熵（内置 log-softmax）
criterion = nn.CrossEntropyLoss()
logits = torch.randn(8, 10)  # batch=8, 10类
labels = torch.randint(0, 10, (8,))
loss = criterion(logits, labels)

# Label Smoothing
criterion_ls = nn.CrossEntropyLoss(label_smoothing=0.1)

# Binary Cross Entropy（多标签）
criterion_bce = nn.BCEWithLogitsLoss()
logits_bin = torch.randn(8, 5)
labels_bin = torch.randint(0, 2, (8, 5)).float()
loss_bce = criterion_bce(logits_bin, labels_bin)
```

### Focal Loss 实现

```python
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p) = -α(1-p)^γ · log(p)
    α: 平衡正负样本权重（正样本常用 0.25）
    γ: 聚焦参数（常用 2.0），越大越关注难样本
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # 计算调制因子
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()
```

### 回归损失

```python
# L1 / L2 / Smooth L1
l1  = nn.L1Loss()
mse = nn.MSELoss()
smooth_l1 = nn.SmoothL1Loss(beta=1.0)  # beta 控制 L1/L2 切换点

pred = torch.randn(4, 4)   # 预测框
gt   = torch.randn(4, 4)   # 真实框
print(smooth_l1(pred, gt))
```

---

## Focal Loss 超参 α 和 γ 调参建议

| 参数 | 含义 | 推荐范围 | 典型值 |
|------|------|----------|--------|
| α | 正样本权重（平衡正负类比例）| 0.1 ~ 0.5 | 0.25（正样本少时用小 α）|
| γ | 聚焦难样本程度 | 0.5 ~ 5.0 | 2.0 |

**调参策略**：
- 正负样本比 1:1 → α ≈ 0.5（不需要平衡）
- 正负样本比 1:100（如目标检测）→ α ≈ 0.25，γ ≈ 2.0
- γ 越大，难样本权重越高，但过大可能导致训练不稳定
- **先固定 γ=2，调 α；再固定 α 调 γ**

---

## IoU 系列损失对比

| 损失 | 公式/特性 | 优缺点 |
|------|-----------|--------|
| IoU Loss | `L = 1 - IoU` | 不重叠时梯度为 0 |
| GIoU Loss | `L = 1 - IoU + |C∖(A∪B)|/|C|`（C 为最小外接框）| 不重叠时有梯度，但收敛慢 |
| DIoU Loss | `L = 1 - IoU + d²/c²`（d 为中心距离，c 为对角线长）| 加速中心点收敛 |
| CIoU Loss | `L = DIoU + αv`（v 衡量宽高比一致性）| 同时优化中心/尺寸/形状，YOLOv5/v8 默认 |

**CIoU 公式**：

```
v = (4/π²)(arctan(w_gt/h_gt) - arctan(w/h))²
α = v / (1 - IoU + v)
CIoU = 1 - IoU + ρ²(b,b_gt)/c² + α·v
```

```python
def ciou_loss(pred_boxes, target_boxes):
    """
    pred_boxes, target_boxes: (N, 4) 格式为 (x1, y1, x2, y2)
    """
    # 使用 torchvision 或 ultralytics 内置实现
    from torchvision.ops import complete_box_iou_loss
    return complete_box_iou_loss(pred_boxes, target_boxes, reduction='mean')
```

---

## 对比学习损失

### InfoNCE（用于自监督学习，如 MoCo）

```
L_InfoNCE = -log( exp(q·k+/τ) / Σ_{i=0}^{K} exp(q·k_i/τ) )
```

- `q`：查询特征，`k+`：正样本键，`k_i`：负样本键
- τ（temperature）：控制分布峰度，通常为 0.07~0.2

```python
def info_nce_loss(q, k_pos, k_neg, temperature=0.07):
    """
    q:     (B, D) 查询特征（已 L2 归一化）
    k_pos: (B, D) 正样本特征
    k_neg: (B*K, D) 负样本特征（来自队列或其他样本）
    """
    q = F.normalize(q, dim=1)
    k_pos = F.normalize(k_pos, dim=1)
    k_neg = F.normalize(k_neg, dim=1)

    # 正样本相似度：(B,)
    pos_sim = (q * k_pos).sum(dim=1, keepdim=True) / temperature
    # 负样本相似度：(B, K)
    neg_sim = (q @ k_neg.T) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)
```

### NT-Xent（用于 SimCLR）

对同一图像的两个增强视图作为正样本对，batch 内其他样本为负样本：

```
L = -log( exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) )
```

---

## 知识蒸馏损失（KL 散度版本）

将教师模型（Teacher）的 soft label 传递给学生模型（Student）：

```
L_distill = KL(T(x)/τ || S(x)/τ) · τ²
L_total = α · L_CE(S(x), y) + (1-α) · L_distill
```

```python
class DistillationLoss(nn.Module):
    """
    temperature τ：软化概率分布，越大越平滑（通常 3~5）
    alpha：蒸馏损失权重（通常 0.9）
    """
    def __init__(self, temperature=4.0, alpha=0.9):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        # 蒸馏损失：KL 散度
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.T ** 2)
        # 标准交叉熵损失
        ce_loss = F.cross_entropy(student_logits, labels)
        return self.alpha * distill_loss + (1 - self.alpha) * ce_loss
```

---

## 分割损失详解

### Dice Loss

```
Dice = 2|A∩B| / (|A| + |B|)
L_Dice = 1 - Dice
```

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

# 实践中常用 BCE + Dice 组合
def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = DiceLoss()(pred, target)
    return 0.5 * bce + 0.5 * dice
```
