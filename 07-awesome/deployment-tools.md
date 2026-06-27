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
