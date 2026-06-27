# TensorRT 部署

> NVIDIA GPU 推理加速框架，相比 PyTorch 推理通常快 2-5 倍。

## 流程

```
PyTorch 模型 → ONNX → TensorRT Engine（.trt）→ 推理
```

## 导出 ONNX

```python
import torch
model.eval()
dummy_input = torch.randn(1, 3, 640, 640).cuda()
torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=17,
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={"images": {0: "batch"}}
)
```

## ONNX → TRT Engine

```bash
# 使用 trtexec（TensorRT 自带工具）
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16 \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:4x3x640x640 \
        --maxShapes=images:8x3x640x640
```

## Python 推理

```python
import tensorrt as trt
import pycuda.driver as cuda

# 加载 engine，分配显存，推理...
```

## 精度模式

| 模式 | 精度损失 | 加速比 |
|------|---------|--------|
| FP32 | 无 | 1× |
| FP16 | 极小 | ~2× |
| INT8 | 需校准数据集 | ~4× |

## 推荐工具

- **torch2trt**：PyTorch 直接转 TRT（简单模型）
- **TensorRT-LLM**：LLM 部署
- **YOLO + TRT**：ultralytics 原生支持 `model.export(format="engine")`
