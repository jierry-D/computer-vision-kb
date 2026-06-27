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
