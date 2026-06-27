# 精选部署工具

## 推理框架

| 工具 | 平台 | 特点 |
|------|------|------|
| TensorRT | NVIDIA GPU | 最快，NVIDIA 官方 |
| ONNX Runtime | 多平台 | 跨平台，CPU/GPU |
| OpenVINO | Intel CPU/iGPU | Intel 优化，工业 PC |
| ncnn | ARM/移动端 | 腾讯出品，手机部署 |
| MNN | 移动端 | 阿里，端侧推理 |
| TFLite | 移动端 | Google，TF 生态 |
| CoreML | Apple | iOS/macOS 硬件加速 |

## 模型压缩

| 工具 | 用途 |
|------|------|
| [torch.quantization](https://pytorch.org/docs/stable/quantization.html) | PyTorch 官方量化 |
| [NNCF](https://github.com/openvinotoolkit/nncf) | Intel，量化+剪枝 |
| [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) | 百度，全套压缩工具 |

## 服务化部署

| 工具 | 特点 |
|------|------|
| [Triton Inference Server](https://github.com/triton-inference-server/server) | NVIDIA，生产级多模型服务 |
| [TorchServe](https://github.com/pytorch/serve) | PyTorch 官方服务框架 |
| [BentoML](https://github.com/bentoml/BentoML) | 快速打包上线 |

---

## 各推理框架性能对比

以 ResNet-50 在同一 GPU（NVIDIA RTX 3090）上的推理速度为参考（batch=1）：

| 框架 | 精度 | 延迟（ms） | 吞吐量（img/s） | 备注 |
|------|------|-----------|-----------------|------|
| PyTorch（FP32） | FP32 | ~8.5 ms | ~118 | 基准 |
| ONNX Runtime（FP32） | FP32 | ~6.0 ms | ~167 | CUDAExecutionProvider |
| TensorRT FP32 | FP32 | ~4.2 ms | ~238 | 1.7× 加速 |
| TensorRT FP16 | FP16 | ~2.1 ms | ~476 | 4× 加速 |
| TensorRT INT8 | INT8 | ~1.3 ms | ~769 | 6× 加速 |

⚠️ **注意**：以上数据为参考量级，实际性能与具体模型结构、输入尺寸、GPU 型号强相关。YOLOv8 等检测模型的加速比通常更高（TRT FP16 vs PyTorch 可达 3-8×）。

| 框架 | 延迟（CPU, Intel i9）| 适用场景 |
|------|---------------------|----------|
| PyTorch | ~120 ms | 开发调试 |
| ONNX Runtime（CPU） | ~85 ms | 通用 CPU 部署 |
| OpenVINO | ~45 ms | Intel CPU 优化部署 |

---

## TorchScript 使用场景与代码

TorchScript 将 PyTorch 模型转为静态图，适合不想引入 ONNX 的场景（纯 PyTorch C++ 部署）。

### 适用场景

- 需要 C++ 推理但不想用 ONNX
- 模型含有 Python 控制流（if/for），ONNX 难以表示
- 与 LibTorch（PyTorch C++ 库）配合使用

```python
import torch

# 方式一：torch.jit.trace（推荐，无控制流时）
model = MyModel()
model.eval()
example_input = torch.randn(1, 3, 640, 640)
traced = torch.jit.trace(model, example_input)

# 保存
traced.save("model_traced.pt")

# 加载推理
loaded = torch.jit.load("model_traced.pt")
with torch.no_grad():
    output = loaded(example_input.cuda())

# 方式二：torch.jit.script（有控制流时）
@torch.jit.script
def my_func(x: torch.Tensor) -> torch.Tensor:
    if x.sum() > 0:
        return x * 2
    else:
        return x + 1

# 对整个模型 script
scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")
```

### C++ 部署代码示意

```cpp
// main.cpp
#include <torch/script.h>
torch::jit::script::Module module = torch::jit::load("model_traced.pt");
module.to(torch::kCUDA);
module.eval();

auto input = torch::randn({1, 3, 640, 640}).to(torch::kCUDA);
std::vector<torch::jit::IValue> inputs = {input};
auto output = module.forward(inputs).toTensor();
```

---

## BentoML 快速部署代码示例

BentoML 让模型服务化只需几行代码，适合快速原型和内部工具：

```bash
pip install bentoml
```

```python
# service.py
import bentoml
import numpy as np
from PIL import Image

# 1. 保存模型到 BentoML 模型仓库
import torch
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

# 保存（自动管理版本）
bentoml.pytorch.save_model("resnet50", model,
                            signatures={"__call__": {"batchable": True}})

# 2. 定义服务
@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 10},
)
class ImageClassifier:
    model_ref = bentoml.models.get("resnet50:latest")

    def __init__(self):
        import torch
        self.model = bentoml.pytorch.load_model(self.model_ref)
        self.model.eval().cuda()

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @bentoml.api
    def classify(self, image: Image.Image) -> dict:
        """接受 PIL Image，返回 Top-5 预测"""
        import torch
        tensor = self.transform(image).unsqueeze(0).cuda()
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=-1)[0]

        top5_probs, top5_idx = probs.topk(5)
        return {
            "predictions": [
                {"class_id": int(idx), "probability": float(prob)}
                for idx, prob in zip(top5_idx, top5_probs)
            ]
        }
```

```bash
# 3. 本地启动服务
bentoml serve service:ImageClassifier --reload

# 4. 测试（另一个终端）
curl -X POST http://localhost:3000/classify \
     -H "Content-Type: multipart/form-data" \
     -F "image=@cat.jpg"

# 5. 打包为容器（生产部署）
bentoml build
bentoml containerize image_classifier:latest
docker run --gpus all -p 3000:3000 image_classifier:latest
```

---

## 模型服务化完整架构

生产环境的模型服务不只是"跑起来"，需要考虑以下完整架构：

```
客户端请求
    │
    ▼
负载均衡层（Nginx / API Gateway）
    │
    ├─→ 限流 / 鉴权 / 路由
    │
    ▼
模型服务层
    ├─ 前处理 Worker（图像解码 / 归一化 / Resize）
    │      │
    │      ▼
    ├─ 推理引擎（TensorRT Engine / Triton Server）
    │      │ batch 聚合（Dynamic Batching）
    │      ▼
    └─ 后处理 Worker（NMS / 解码 / 格式化）
    │
    ▼
缓存层（Redis，缓存相同请求结果）
    │
    ▼
结果返回给客户端
    │
    └─→ 日志 / 监控（Prometheus + Grafana）
         ├─ 延迟 P99 / P50
         ├─ 每秒请求数（QPS）
         ├─ GPU 利用率
         └─ 异常检测（结果置信度分布异常告警）
```

**关键优化点**：
1. **Dynamic Batching**：将多个单张请求聚合为 batch 推理，显著提升 GPU 利用率
2. **模型预热**：服务启动时先推理几次，避免首次请求延迟高
3. **异步处理**：前/后处理与 GPU 推理并行，隐藏 CPU 计算延迟
4. **多实例**：单机多 GPU 时，每块 GPU 独立服务实例，通过负载均衡分发

💡 **Tips**：小型项目（QPS < 100）用 BentoML 或 FastAPI 包装即可；中大型项目（QPS > 1000）推荐使用 Triton Inference Server，内置 Dynamic Batching 和多模型管理。
