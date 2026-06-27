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

---

## TRT Python API 完整推理代码

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化 CUDA context
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    """从文件加载 TensorRT engine"""
    with open(engine_path, "rb") as f, \
         trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

class TRTInferencer:
    def __init__(self, engine_path):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # 分配显存（host + device）
        self.inputs  = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))

            # 分配 host（CPU）和 device（GPU）内存
            host_mem   = cuda.pagelocked_empty(np.prod(shape), dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem,
                                    'name': name, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem,
                                     'name': name, 'shape': shape})

        self.stream = cuda.Stream()

    def infer(self, input_array):
        """
        执行推理
        input_array: numpy array，形状与 engine 输入一致
        """
        # 拷贝输入到显存
        np.copyto(self.inputs[0]['host'], input_array.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'],
                               self.inputs[0]['host'], self.stream)

        # 执行推理
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 拷贝输出到内存
        results = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()

        for out in self.outputs:
            results.append(out['host'].reshape(out['shape']))
        return results

    def __del__(self):
        del self.context
        del self.engine

# 使用示例
inferencer = TRTInferencer("model.trt")
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = inferencer.infer(input_data)
print(f"输出 shape: {[o.shape for o in outputs]}")
```

---

## INT8 量化：校准集准备与校准代码

INT8 量化需要"校准"步骤来确定每层的量化范围，需要准备 100-500 张代表性图像。

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import os
from PIL import Image

class ImageBatchStream:
    """用于 INT8 量化的图像 Batch 迭代器"""
    def __init__(self, image_dir, batch_size=8, input_size=(640, 640)):
        self.batch_size = batch_size
        self.input_size = input_size
        self.image_files = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ][:500]  # 最多 500 张
        self.batch_idx = 0

    def __len__(self):
        return (len(self.image_files) + self.batch_size - 1) // self.batch_size

    def next_batch(self):
        start = self.batch_idx * self.batch_size
        end   = min(start + self.batch_size, len(self.image_files))
        if start >= len(self.image_files):
            return None

        batch = np.zeros((self.batch_size, 3, *self.input_size), dtype=np.float32)
        for i, path in enumerate(self.image_files[start:end]):
            img = Image.open(path).convert('RGB').resize(self.input_size)
            img_arr = np.array(img).astype(np.float32) / 255.0
            img_arr = (img_arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            batch[i] = img_arr.transpose(2, 0, 1)

        self.batch_idx += 1
        return batch

class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """熵校准器（推荐，比 MinMax 精度更好）"""
    def __init__(self, stream, cache_file="calibration.cache"):
        super().__init__()
        self.stream = stream
        self.cache_file = cache_file
        self.device_input = cuda.mem_alloc(
            stream.batch_size * 3 * 640 * 640 * 4  # float32 = 4 bytes
        )

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        batch = self.stream.next_batch()
        if batch is None:
            return None
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# 构建 INT8 Engine
def build_int8_engine(onnx_path, calibration_images, output_engine):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    config.set_flag(trt.BuilderFlag.INT8)

    # 设置校准器
    stream = ImageBatchStream(calibration_images)
    calibrator = Int8EntropyCalibrator(stream)
    config.int8_calibrator = calibrator

    engine = builder.build_serialized_network(network, config)
    with open(output_engine, "wb") as f:
        f.write(engine)
    print(f"INT8 Engine 已保存到 {output_engine}")

build_int8_engine("model.onnx", "/data/calibration/", "model_int8.trt")
```

---

## TRT 性能 Benchmark 方法

```bash
# 使用 trtexec 进行标准 Benchmark（推荐）
trtexec --loadEngine=model.trt \
        --iterations=1000 \           # 推理次数
        --warmUp=500 \                # 热身次数（ms）
        --duration=10 \               # 测试持续时间（s）
        --avgRuns=10 \                # 平均次数
        --percentile=99               # 报告 P99 延迟

# FP16 Benchmark（构建 + 测试）
trtexec --onnx=model.onnx \
        --fp16 \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:1x3x640x640 \
        --maxShapes=images:1x3x640x640 \
        --iterations=300

# 输出示例：
# [I] Latency: min = 2.45 ms, max = 3.12 ms, mean = 2.61 ms, median = 2.58 ms, percentile(99%) = 3.09 ms
# [I] Throughput: 383.5 qps
```

---

## 常见报错与解决方法

| 报错信息 | 原因 | 解决方案 |
|----------|------|----------|
| `[TRT] [E] ... Unsupported ONNX opset version` | ONNX opset 版本过高 | TRT10 支持到 opset 20，降低 opset 版本 |
| `[TRT] [E] ... No implementation for layer` | 算子不支持 | 检查 TRT 版本，升级或用支持算子替换 |
| `[TRT] [E] INVALID_CONFIG: ... dynamic shapes` | 动态 shape 配置错误 | 导出时设置 `dynamic_axes`，构建时设置 `OptimizationProfile` |
| `pycuda._driver.LogicError: cuLaunchKernel failed` | CUDA context 问题 | 确保同一线程创建和使用 context，多线程需绑定 context |
| `Segmentation fault` 推理时崩溃 | 显存越界 | 检查输出 tensor 分配大小，确认 shape 计算正确 |
| INT8 精度大幅下降 | 校准数据不具代表性 | 增加校准数据量，确保覆盖测试场景的分布 |
| 构建时间极长（>1小时）| 大 batch + 多动态维度 | 减少动态维度数量，固定 batch=1 |

---

## TRT 与 PyTorch 推理结果对比验证代码

部署后必须验证 TRT 结果与 PyTorch 结果的一致性：

```python
import torch
import numpy as np
from inferencer import TRTInferencer  # 上面定义的类

def compare_trt_pytorch(pytorch_model, trt_engine_path, test_input):
    """
    对比 PyTorch 和 TRT 推理结果
    test_input: numpy array [N, C, H, W]
    """
    # PyTorch 推理
    pytorch_model.eval().cuda()
    with torch.no_grad():
        torch_input = torch.from_numpy(test_input).cuda()
        torch_output = pytorch_model(torch_input).cpu().numpy()

    # TRT 推理
    trt_infer = TRTInferencer(trt_engine_path)
    trt_output = trt_infer.infer(test_input)[0]

    # 对比指标
    abs_diff = np.abs(torch_output - trt_output)
    rel_diff = abs_diff / (np.abs(torch_output) + 1e-8)

    print("=== TRT vs PyTorch 推理对比 ===")
    print(f"输出 shape: PyTorch={torch_output.shape}, TRT={trt_output.shape}")
    print(f"绝对误差: max={abs_diff.max():.6f}, mean={abs_diff.mean():.6f}")
    print(f"相对误差: max={rel_diff.max():.4%}, mean={rel_diff.mean():.4%}")
    print(f"最大偏差位置: {np.unravel_index(abs_diff.argmax(), abs_diff.shape)}")

    # 判断是否可接受（FP16 下通常 < 1%，INT8 下可能 < 5%）
    threshold = 0.01  # FP16
    passed = rel_diff.mean() < threshold
    print(f"验证{'通过' if passed else '失败'} (阈值={threshold:.1%})")
    return passed

# 使用
import random
test_images = np.random.randn(4, 3, 640, 640).astype(np.float32)
compare_trt_pytorch(model, "model_fp16.trt", test_images)
```
