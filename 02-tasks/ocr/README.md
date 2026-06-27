# OCR（文字检测与识别）

## 任务流程

```
图像 → 文字检测（定位文字区域）→ 文字识别（转为文本）→ 后处理
```

## 文字检测

| 模型 | 特点 |
|------|------|
| EAST | 旋转框检测，速度快 |
| DBNet | 可微分二值化，精度高 |
| DBNet++ | 改进版，工业常用 |
| TextBPN | 任意形状文字 |

## 文字识别

| 模型 | 特点 |
|------|------|
| CRNN | CNN + BiLSTM + CTC，经典 |
| ASTER | 矫正 + 注意力识别 |
| ABINet | 语言模型辅助 |
| PARSeq | Permutation 自回归，SOTA |

## 端到端 OCR 框架

| 框架 | 特点 |
|------|------|
| PaddleOCR | 百度出品，中文最强，工业首选 |
| EasyOCR | 80+ 语言，开箱即用 |
| TrOCR | Transformer，微软 |
| Tesseract | 传统方案，多语言 |

## 快速上手（PaddleOCR）

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')
result = ocr.ocr('image.jpg', cls=True)
```

---

## DBNet：可微分二值化

传统文字检测后处理（Hough 变换/二值化）不可微，无法端到端训练。DBNet 提出**可微分二值化（DB）**，使二值化过程参与梯度反传。

### 核心原理

```python
import torch
import torch.nn as nn

class DifferentiableBinarization(nn.Module):
    def __init__(self, k=50):
        super().__init__()
        self.k = k  # 放大系数，越大越接近硬阈值

    def forward(self, prob_map, threshold_map):
        """
        prob_map:      [B, 1, H, W]，预测的文字概率图
        threshold_map: [B, 1, H, W]，自适应阈值图
        返回:          [B, 1, H, W]，近似二值化结果
        """
        # 标准二值化：B(x) = 1 if P(x) > T(x) else 0
        # 可微分近似：B(x) ≈ sigmoid(k * (P(x) - T(x)))
        binary_map = torch.sigmoid(self.k * (prob_map - threshold_map))
        return binary_map

# DBNet 预测三个头
class DBHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.prob_head  = nn.Conv2d(in_channels, 1, 1)   # 概率图
        self.thresh_head = nn.Conv2d(in_channels, 1, 1)  # 阈值图
        self.db = DifferentiableBinarization(k=50)

    def forward(self, x):
        prob  = torch.sigmoid(self.prob_head(x))
        thresh = torch.sigmoid(self.thresh_head(x))
        binary = self.db(prob, thresh)
        return prob, thresh, binary
```

### 后处理（Polygon 提取）

```python
import cv2
import numpy as np

def db_postprocess(binary_map, min_area=16, unclip_ratio=1.5):
    """从二值化图提取文字区域多边形"""
    _, thresh = cv2.threshold(binary_map, 0.3, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        # 用 Polygon 扩张（unclip）扩大文字框范围
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        expand = unclip_ratio * (w * h) / (2 * (w + h) + 1e-9)
        box = cv2.boxPoints(rect)
        # 简单扩张：沿法向量扩展 expand 像素
        boxes.append(box)
    return boxes
```

---

## CRNN：网络结构详解

CRNN（Convolutional Recurrent Neural Network）是文字识别的经典架构：

```
输入图像 [B, 1, H, W]（灰度，高度统一为 32px）
   ↓ CNN 特征提取（VGG-like，最终高度压缩为 1）
   → 特征序列 [B, C, 1, W/4]，reshape 为 [W/4, B, C]
   ↓ BiLSTM × 2（时序建模）
   → [W/4, B, hidden_size*2]
   ↓ 全连接层 → vocab_size + 1（+1 为 CTC blank）
   ↓ CTC 解码 → 文字序列
```

```python
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_h=32, nc=1, nclass=37, nh=256):
        super().__init__()
        # CNN 骨干：将 [B, 1, 32, W] 压缩为 [B, 512, 1, W/4]
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # H/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # H/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),                # H/8，W 不变
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),                # H/16
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(),
            # 最终输出 [B, 512, 1, W/4]
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, x):
        conv = self.cnn(x)               # [B, 512, 1, T]
        conv = conv.squeeze(2)           # [B, 512, T]
        conv = conv.permute(2, 0, 1)     # [T, B, 512]
        return self.rnn(conv)            # [T, B, nclass]

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=False)
        self.fc  = nn.Linear(hidden_size * 2, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)
```

---

## CTC 解码：Greedy 与 Beam Search

```python
import torch

def ctc_greedy_decode(log_probs, blank=0):
    """
    log_probs: [T, B, C]（对数概率）
    blank:     CTC blank 标签 ID
    """
    probs = log_probs.exp()
    pred_ids = probs.argmax(dim=2)  # [T, B]
    results = []
    for b in range(pred_ids.shape[1]):
        seq = pred_ids[:, b].tolist()
        # 去重（合并相邻重复），去 blank
        decoded = []
        prev = -1
        for s in seq:
            if s != blank and s != prev:
                decoded.append(s)
            prev = s
        results.append(decoded)
    return results

# Beam Search CTC（使用 ctcdecode 库）
def ctc_beam_search_decode(log_probs, beam_width=10, blank=0):
    """
    需要安装: pip install ctcdecode
    """
    from ctcdecode import CTCBeamDecoder
    # vocab 需要按 ID 顺序排列
    decoder = CTCBeamDecoder(
        vocab,
        beam_width=beam_width,
        blank_id=blank,
        log_probs_input=True
    )
    # log_probs: [B, T, C]
    beam_results, _, _, out_lens = decoder.decode(log_probs.permute(1, 0, 2))
    return [beam_results[b][0][:out_lens[b][0]].tolist() for b in range(len(log_probs))]
```

---

## PaddleOCR 自定义字典训练流程

```bash
# 1. 准备数据（文件格式：图片路径\t标注文本）
# train_list.txt:
# /data/ocr/train/00001.jpg	你好世界
# /data/ocr/train/00002.jpg	OpenCV

# 2. 准备自定义字典（每行一个字符）
echo "你好世界OpenCV..." > my_dict.txt

# 3. 修改配置文件（rec_chinese_common_train_v2.0.yml）
# character_dict_path: my_dict.txt
# use_space_char: true

# 4. 下载预训练模型（迁移学习）
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar
tar xf ch_PP-OCRv3_rec_train.tar

# 5. 启动训练
python tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml \
    -o Global.pretrained_model=./ch_PP-OCRv3_rec_train/best_accuracy \
       Train.dataset.data_dir=/data/ocr/train \
       Train.dataset.label_file_list=train_list.txt \
       Eval.dataset.data_dir=/data/ocr/val \
       Eval.dataset.label_file_list=val_list.txt \
       Global.character_dict_path=my_dict.txt
```

---

## OCR 后处理

### 文字方向检测

```python
from paddleocr import PaddleOCR

# 开启方向分类器（支持 0°/180° 旋转文字）
ocr = PaddleOCR(use_angle_cls=True, lang='ch', cls=True)
result = ocr.ocr('rotated_image.jpg')
for line in result[0]:
    bbox, (text, confidence) = line
    print(f"文字: {text} | 置信度: {confidence:.3f} | 位置: {bbox}")
```

### 版面分析

```python
from paddleocr import PPStructure

# 版面分析：自动识别标题/段落/表格/图片区域
table_engine = PPStructure(table=False, ocr=True, show_log=True)
result = table_engine('document.jpg')
for region in result:
    print(f"区域类型: {region['type']}")  # text/title/figure/table
    if region['type'] == 'text':
        print(f"内容: {region['res']}")
```

---

## 表格识别简介

表格识别分为：**单元格检测** → **内容识别** → **结构还原**

```python
from paddleocr import PPStructure

table_engine = PPStructure(table=True, ocr=True)
result = table_engine('table.jpg')
for region in result:
    if region['type'] == 'table':
        # 输出为 HTML 格式，便于解析
        html = region['res']['html']
        print(html)
```

**踩坑记录**：
- 倾斜文字（> 15°）要先用 text_det + text_cls 矫正再识别，否则 CRNN 类模型准确率大幅下降
- PaddleOCR 默认输入图像最长边 960px，超过会自动缩放，对小字体不友好，可调大 `det_limit_side_len`
- 自定义字典中必须包含训练集中所有字符，漏掉字符会导致该字符无法被识别
