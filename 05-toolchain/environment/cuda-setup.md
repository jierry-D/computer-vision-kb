# CUDA / cuDNN 环境配置

## 版本对应关系

安装前确认版本兼容：[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)

| PyTorch | CUDA | cuDNN |
|---------|------|-------|
| 2.3.x | 12.1 / 11.8 | 8.x |
| 2.1.x | 12.1 / 11.8 | 8.x |
| 2.0.x | 11.7 / 11.8 | 8.x |

## 安装步骤

### 1. 查看驱动支持的最高 CUDA 版本
```bash
nvidia-smi
```

### 2. 安装 PyTorch（推荐 conda）
```bash
# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 或 pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 验证安装
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## 常见问题

- `CUDA out of memory`：减小 batch size，或用梯度累积
- cuDNN 版本不匹配：严格按 PyTorch 官网命令安装，不要单独安装 cuDNN
