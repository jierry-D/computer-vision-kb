# ONNX 部署

> Open Neural Network Exchange：跨框架模型交换格式。

## 用途

- PyTorch/TF 模型 → ONNX → TensorRT / OpenVINO / ONNX Runtime
- 跨平台部署的中间桥梁

## 导出与验证

```python
import torch, onnx, onnxruntime as ort
import numpy as np

# 导出
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=17)

# 验证结构
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# ONNX Runtime 推理
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: np.random.randn(1, 3, 640, 640).astype(np.float32)})
```

## 常见问题

| 问题 | 解决 |
|------|------|
| 算子不支持 | 升级 opset 版本，或自定义算子 |
| 动态 shape | 导出时指定 `dynamic_axes` |
| 精度对不上 | 检查输入归一化是否一致 |

## 优化工具

```bash
pip install onnxsim
python -m onnxsim model.onnx model_sim.onnx  # 简化图结构，加速推理
```
