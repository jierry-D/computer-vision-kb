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

---

## ONNX 模型可视化工具 Netron

Netron 是可视化 ONNX（以及 TensorRT/TFLite 等）模型图结构的最佳工具。

```bash
# 安装
pip install netron

# 启动（会在浏览器中打开）
netron model.onnx
# 或指定端口
netron model.onnx --port 8888
```

也可以直接访问在线版：**netron.app**（直接拖入模型文件，无需安装）

**Netron 的用途**：
- 查看每个算子的类型和参数
- 确认输入输出节点名称（推理时需要）
- 发现不支持的算子（显示为红色警告）
- 对比简化前后的图结构差异

---

## 常见不支持算子的处理方案

### 方案一：升级 opset 版本

```python
# 较新的 opset 版本支持更多算子
torch.onnx.export(model, dummy_input, "model.onnx",
                  opset_version=17)  # 推荐 16 或 17
```

### 方案二：替换不支持的算子（修改模型）

```python
import torch.nn as nn

# 示例：将不支持的激活函数替换为支持的
class DeployModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        # 替换 SiLU → ReLU（TRT 早期版本不支持 SiLU）
        self._replace_activation()

    def _replace_activation(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.SiLU):
                parent = self._get_parent(name)
                attr_name = name.split('.')[-1]
                setattr(parent, attr_name, nn.ReLU(inplace=True))
```

### 方案三：自定义算子注册（高级）

```python
from torch.onnx import register_custom_op_symbolic

# 注册自定义算子的 ONNX 符号定义
def custom_op_symbolic(g, input, scale):
    return g.op("custom_domain::CustomOp", input, scale_f=scale)

register_custom_op_symbolic("custom_ops::custom_op", custom_op_symbolic, opset_version=11)
```

### 常见不支持算子及解决方案

| 算子 | 问题场景 | 解决方案 |
|------|----------|----------|
| `grid_sample` | STN / 可变形卷积 | 升级 opset 到 16+ |
| `scatter_nd` | 3D 检测头 | 用等价操作替换 |
| `einsum` | Transformer 注意力 | 用 `bmm` 手动实现 |
| `LayerNorm` | Transformer | 升级 opset 到 17+ |
| `Mish/SiLU` | YOLO 系列 | 替换为 `ReLU` 或 `GELU` |

---

## 动态 shape 推理完整代码

```python
import torch
import onnxruntime as ort
import numpy as np

# === 导出动态 shape ONNX ===
model = ...  # 你的 PyTorch 模型
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model,
    dummy_input,
    "model_dynamic.onnx",
    opset_version=17,
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={
        "images": {
            0: "batch_size",    # batch 维度动态
            2: "height",        # 高度动态
            3: "width",         # 宽度动态
        },
        "output": {
            0: "batch_size",
        }
    }
)

# === 动态 shape 推理 ===
# 创建会话时指定优化级别
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "model_dynamic.onnx",
    sess_options=options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# 打印模型输入输出信息
for inp in session.get_inputs():
    print(f"输入: {inp.name}, shape={inp.shape}, dtype={inp.type}")
for out in session.get_outputs():
    print(f"输出: {out.name}, shape={out.shape}, dtype={out.type}")

# 使用不同尺寸推理
for h, w in [(320, 320), (640, 640), (1280, 720)]:
    input_data = np.random.randn(1, 3, h, w).astype(np.float32)
    outputs = session.run(None, {"images": input_data})
    print(f"输入 {h}×{w} → 输出 shape: {[o.shape for o in outputs]}")
```

---

## ONNX 模型量化（onnxruntime 量化 API）

```python
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
    QuantType
)
import numpy as np

# === 方式一：动态量化（无需校准数据，速度快，适合 CPU 部署）===
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quant_dynamic.onnx",
    weight_type=QuantType.QInt8,     # 权重量化为 INT8
)
print("动态量化完成")

# === 方式二：静态量化（需要校准数据，精度更好）===
class ImageCalibrationReader(CalibrationDataReader):
    """准备校准数据集（通常用 100-200 张有代表性的图）"""
    def __init__(self, image_folder, input_name="images"):
        import os
        from PIL import Image
        self.input_name = input_name
        self.data = []

        for fname in os.listdir(image_folder)[:200]:
            img = Image.open(os.path.join(image_folder, fname)).resize((640, 640))
            img_arr = np.array(img).astype(np.float32) / 255.0
            img_arr = img_arr.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW
            # ImageNet 归一化
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std  = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            img_arr = (img_arr - mean) / std
            self.data.append({self.input_name: img_arr.astype(np.float32)})

        self.enum_data = iter(self.data)

    def get_next(self):
        return next(self.enum_data, None)

# 预处理原始模型（为量化做准备）
from onnxruntime.quantization import preprocess
preprocess.quant_pre_process("model.onnx", "model_preprocessed.onnx")

# 静态量化
quantize_static(
    model_input="model_preprocessed.onnx",
    model_output="model_quant_static.onnx",
    calibration_data_reader=ImageCalibrationReader("/path/to/calibration/images"),
    quant_format=QuantType.QInt8,
)
print("静态量化完成")
```

---

## 多输入多输出模型推理代码

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("multi_io_model.onnx",
                                providers=["CUDAExecutionProvider"])

# 查看所有输入
print("模型输入：")
for inp in session.get_inputs():
    print(f"  名称: {inp.name}, 形状: {inp.shape}, 类型: {inp.type}")

print("模型输出：")
for out in session.get_outputs():
    print(f"  名称: {out.name}, 形状: {out.shape}, 类型: {out.type}")

# 准备多输入（例如：图像 + 文本特征 + 先验框）
feeds = {
    "image":     np.random.randn(1, 3, 640, 640).astype(np.float32),
    "text_feat": np.random.randn(1, 512).astype(np.float32),
    "anchors":   np.random.randn(1, 8400, 4).astype(np.float32),
}

# 只获取需要的输出（提升效率）
output_names = ["boxes", "scores"]  # 仅返回这两个输出
outputs = session.run(output_names, feeds)

boxes, scores = outputs
print(f"boxes shape: {boxes.shape}")
print(f"scores shape: {scores.shape}")

# 批量推理
batch_results = []
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    out = session.run(None, {"image": batch})
    batch_results.extend(out[0])
```

💡 **Tips**：推理时建议将数组预先分配在 `numpy` 中并复用（避免每次推理都重新分配内存），可进一步提升吞吐量。
