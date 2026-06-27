# 人脸视觉

## 子任务

| 任务 | 说明 |
|------|------|
| 人脸检测 | 定位图像中的人脸框 |
| 人脸对齐 | 检测关键点，矫正为正脸 |
| 人脸识别 | 身份验证（1:1）/ 搜索（1:N） |
| 人脸属性 | 年龄/性别/表情/戴眼镜等 |
| 人脸生成 | GAN/Diffusion 生成或编辑 |

## 人脸检测

| 模型 | 特点 |
|------|------|
| RetinaFace | 精度高，关键点同步检测 |
| SCRFD | 速度快，适合边缘部署 |
| YOLOv8-face | YOLO 系列定制版 |

## 人脸识别

| 模型/方法 | 特点 |
|---------|------|
| ArcFace | Additive Angular Margin Loss，主流 |
| CosFace | Large Margin Cosine Loss |
| AdaFace | 自适应 margin，低质量图像强 |

### 常用流程

```python
# 1. 检测 + 对齐
# 2. 特征提取（512-d embedding）
# 3. 余弦相似度比较
similarity = F.cosine_similarity(feat1, feat2)
```

## 数据集

- **训练**：MS1MV2（5.8M 图，85k 人）
- **测试**：LFW、CFP-FP、IJB-C

---

## RetinaFace：多任务联合学习

RetinaFace 是人脸检测的工业标准，同时预测：
- 人脸检测框（bbox）
- 5 个关键点（双眼/鼻尖/嘴角，用于对齐）
- 3D 人脸重建（dense landmark，可选）

```python
from retinaface import RetinaFace

# 快速检测
faces = RetinaFace.detect_faces('group.jpg')
for key, face_info in faces.items():
    bbox       = face_info['facial_area']     # [x1, y1, x2, y2]
    landmarks  = face_info['landmarks']        # {'right_eye': (x,y), ...}
    confidence = face_info['score']
    print(f"{key}: bbox={bbox}, conf={confidence:.3f}")

# 带对齐的裁剪（对齐为 112×112 正脸）
aligned_faces = RetinaFace.extract_faces('group.jpg', align=True)
```

### 关键点对齐（仿射变换到标准脸）

```python
import cv2
import numpy as np

# 112×112 标准人脸关键点模板（ArcFace 规范）
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],  # 右眼
    [73.5318, 51.5014],  # 左眼
    [56.0252, 71.7366],  # 鼻尖
    [41.5493, 92.3655],  # 右嘴角
    [70.7299, 92.2041],  # 左嘴角
], dtype=np.float32)

def align_face(image, landmarks, output_size=(112, 112)):
    """
    landmarks: {'right_eye': (x,y), 'left_eye': ..., 'nose': ...,
                'mouth_right': ..., 'mouth_left': ...}
    """
    src = np.array([
        landmarks['right_eye'],
        landmarks['left_eye'],
        landmarks['nose'],
        landmarks['mouth_right'],
        landmarks['mouth_left'],
    ], dtype=np.float32)

    M = cv2.estimateAffinePartial2D(src, ARCFACE_TEMPLATE)[0]
    aligned = cv2.warpAffine(image, M, output_size,
                              borderValue=(0, 0, 0))
    return aligned
```

---

## ArcFace 损失完整实现

ArcFace 在余弦相似度基础上增加角度 margin，使同类特征更聚类、异类更分散：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    公式: logit_yi = s * cos(θ_yi + m)，其他类 = s * cos(θ_j)
    """
    def __init__(self, in_features, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s = s          # 缩放因子，通常为 32 或 64
        self.m = m          # 角度 margin（弧度），通常 0.5
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # 预计算辅助值
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # cos(π - m) 用于判断是否在安全区间
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        """
        embeddings: [B, D]，L2 归一化后的特征向量
        labels:     [B]，类别索引
        """
        # 1. L2 归一化权重和特征
        embeddings = F.normalize(embeddings, dim=1)
        W = F.normalize(self.weight, dim=1)

        # 2. 计算余弦相似度：cos(θ)
        cos_theta = F.linear(embeddings, W)          # [B, num_classes]
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

        # 3. 计算目标类的 cos(θ + m)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # 4. 处理 θ + m > π 的情况（避免梯度消失）
        cos_theta_m = torch.where(
            cos_theta > self.th,
            cos_theta_m,
            cos_theta - self.mm
        )

        # 5. 将目标类 logit 替换为 cos(θ + m)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * cos_theta_m + (1 - one_hot) * cos_theta
        logits = logits * self.s

        return F.cross_entropy(logits, labels)
```

---

## 人脸识别系统完整流程

```python
import numpy as np
import torch
import torch.nn.functional as F
from retinaface import RetinaFace

class FaceRecognitionSystem:
    def __init__(self, recognition_model, threshold=0.4):
        """
        recognition_model: 输入 [B, 3, 112, 112]，输出 [B, 512] embedding
        threshold:         余弦相似度阈值（>= 则认定为同一人）
        """
        self.model = recognition_model.eval()
        self.threshold = threshold
        self.gallery = {}  # 人员库 {name: embedding}

    def extract_embedding(self, aligned_face):
        """提取单张对齐人脸的特征向量"""
        img_tensor = preprocess(aligned_face)  # → [1, 3, 112, 112]
        with torch.no_grad():
            emb = self.model(img_tensor)
        return F.normalize(emb, dim=1).squeeze(0)  # [512]

    def register(self, name, image_path):
        """注册人员到库"""
        faces = RetinaFace.detect_faces(image_path)
        if not faces:
            raise ValueError(f"未检测到人脸：{image_path}")
        face_info = list(faces.values())[0]
        img = cv2.imread(image_path)
        aligned = align_face(img, face_info['landmarks'])
        self.gallery[name] = self.extract_embedding(aligned)
        print(f"注册成功：{name}")

    def recognize(self, image_path, top_k=1):
        """识别图像中的人脸"""
        faces = RetinaFace.detect_faces(image_path)
        img = cv2.imread(image_path)
        results = []

        for face_info in faces.values():
            aligned = align_face(img, face_info['landmarks'])
            query_emb = self.extract_embedding(aligned)

            # 与所有注册人员计算余弦相似度
            similarities = {
                name: F.cosine_similarity(query_emb.unsqueeze(0),
                                          emb.unsqueeze(0)).item()
                for name, emb in self.gallery.items()
            }
            # 按相似度降序排列
            ranked = sorted(similarities.items(), key=lambda x: -x[1])
            best_name, best_score = ranked[0]

            if best_score >= self.threshold:
                results.append({'name': best_name, 'score': best_score,
                                'bbox': face_info['facial_area']})
            else:
                results.append({'name': '陌生人', 'score': best_score,
                                'bbox': face_info['facial_area']})
        return results
```

---

## 活体检测（Face Anti-Spoofing）

活体检测用于防御照片/视频/3D 模具攻击：

| 方法 | 原理 | 优缺点 |
|------|------|--------|
| **纹理分析** | 区分真实皮肤与打印纹理 | 简单，易被高清打印欺骗 |
| **深度估计** | 真实人脸有三维深度 | 效果好，需辅助传感器 |
| **rPPG** | 检测皮肤血流脉冲信号 | 难以伪造，计算量大 |
| **Vision Transformer** | 多帧特征提取 | 当前 SOTA，MiniFASNet 等 |

```python
# 使用 Silent-Face-Anti-Spoofing 库
# pip install silent-face-anti-spoofing
from src.anti_spoof_predict import AntiSpoofPredict

model = AntiSpoofPredict(device_id=0)

def check_liveness(image, bbox):
    """
    bbox: [x1, y1, x2, y2, score]
    返回: (is_real, confidence)
    """
    prediction = model.predict(image, bbox)
    label = np.argmax(prediction)
    # label=1: 真实人脸，label=0: 攻击
    return (label == 1), float(prediction[label])
```

---

## 人脸数据隐私合规注意事项

**法规要求（中国）**：
- 《个人信息保护法》：人脸数据属于**敏感个人信息**，须单独授权同意
- 《信息安全技术 人脸识别数据安全要求》：存储需加密，传输需加密
- 禁止在公共场所滥用人脸识别

**技术层面建议**：

1. **特征加密存储**：不直接存储原始人脸图像，只存储加密后的特征向量
2. **差分隐私**：在特征向量上加入受控噪声，防止模型逆向还原
3. **数据最小化**：仅保留业务必需的信息，不存储无关属性
4. **访问控制**：人脸特征库严格权限管控，记录所有访问日志

```python
# 特征向量加密存储示例（AES-256）
from cryptography.fernet import Fernet

def encrypt_embedding(embedding: np.ndarray, key: bytes) -> bytes:
    f = Fernet(key)
    return f.encrypt(embedding.tobytes())

def decrypt_embedding(encrypted: bytes, key: bytes) -> np.ndarray:
    f = Fernet(key)
    return np.frombuffer(f.decrypt(encrypted), dtype=np.float32)
```

**踩坑记录**：
- LFW 测试集准确率 > 99% 不代表真实场景可用，真实场景的低质量图（模糊/遮挡/大姿态）才是瓶颈
- 人脸底库更新（增删人员）时需考虑特征版本兼容性，避免跨模型版本比对
