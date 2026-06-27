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

---

## Docker + CUDA 镜像方案（强烈推荐）

使用 Docker 隔离环境是工业项目和团队合作的最佳实践，避免系统级 CUDA 冲突。

### 推荐基础镜像

```bash
# NVIDIA 官方镜像（nvcr.io/nvidia/pytorch）
# 格式：xx.yy-py3，如 24.01-py3 表示 2024年1月版本
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# 或使用 PyTorch 官方镜像
docker pull pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
```

### 推荐 Dockerfile

```dockerfile
# Dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

# 设置工作目录
WORKDIR /workspace

# 安装常用 CV 工具
RUN pip install --no-cache-dir \
    opencv-python-headless \
    albumentations \
    timm \
    mmengine \
    supervision \
    onnxruntime-gpu \
    tensorboard

# 复制项目文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0
```

```bash
# 构建镜像
docker build -t my-cv-env:latest .

# 运行容器（挂载数据集和代码）
docker run --gpus all \
    -v /data:/data \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    -it my-cv-env:latest bash

# 启动 Jupyter
docker run --gpus all \
    -v /data:/data \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    my-cv-env:latest \
    jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

### Docker Compose 示例

```yaml
# docker-compose.yml
version: '3.8'
services:
  cv-dev:
    image: my-cv-env:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - /data:/data
      - .:/workspace
    ports:
      - "8888:8888"
    ipc: host    # 共享内存，DataLoader 多进程需要
    stdin_open: true
    tty: true
```

---

## 多 CUDA 版本共存管理

有时不同项目需要不同 CUDA 版本，可以用以下方式共存：

### 方案一：update-alternatives 切换（系统级）

```bash
# 安装多个 CUDA 版本后，用 update-alternatives 切换
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-11.8 118
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-12.1 121

# 查看和切换
sudo update-alternatives --config cuda
# 选择对应数字即可切换
```

### 方案二：conda 环境隔离（推荐）

每个 conda 环境安装不同版本的 PyTorch，对应不同 CUDA：

```bash
# 环境1：CUDA 11.8 + PyTorch 2.0
conda create -n pt20-cu118 python=3.10
conda activate pt20-cu118
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# 环境2：CUDA 12.1 + PyTorch 2.3
conda create -n pt23-cu121 python=3.11
conda activate pt23-cu121
conda install pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## Conda 环境隔离完整工作流

```bash
# 1. 创建新环境
conda create -n cv_project python=3.10 -y
conda activate cv_project

# 2. 安装 PyTorch（先装，避免依赖冲突）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 导出环境配置（团队协作用）
conda env export > environment.yml
# 或只导出手动安装的包（更简洁）
conda env export --from-history > environment_min.yml

# 5. 从配置文件恢复环境
conda env create -f environment.yml

# 6. 删除环境（不需要时清理）
conda deactivate
conda env remove -n cv_project
```

```yaml
# environment.yml 示例
name: cv_project
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.3.0
  - torchvision=0.18.0
  - pytorch-cuda=12.1
  - pip:
    - opencv-python
    - albumentations
    - timm
    - mmdet
```

---

## GPU 监控命令

```bash
# 基础监控（NVIDIA 官方）
nvidia-smi                          # 当前状态快照
nvidia-smi -l 2                     # 每 2 秒刷新
watch -n 1 nvidia-smi               # 每秒刷新（更简洁）

# 查看显存使用
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# 进程级显存占用
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# gpustat（推荐，比 nvidia-smi 简洁）
pip install gpustat
gpustat -i 1                        # 每秒刷新
gpustat --show-pid                  # 显示进程信息

# nvitop（最强可视化 GPU 监控工具）
pip install nvitop
nvitop                              # 交互式 TUI 界面
nvitop -m full                      # 完整模式
```

---

## 显存 OOM（Out of Memory）排查流程

遇到 `RuntimeError: CUDA out of memory` 时，按以下步骤系统排查：

### 快速解决

```python
import torch
import gc

# 1. 清理缓存（训练前/每个 epoch 后）
torch.cuda.empty_cache()
gc.collect()

# 2. 查看当前显存使用
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.memory_reserved() / 1e9,  "GB reserved")

# 3. 详细显存报告
print(torch.cuda.memory_summary())
```

### 系统性排查清单

| 问题 | 检查项 | 解决方法 |
|------|--------|----------|
| Batch size 太大 | 减小 batch_size | 使用梯度累积 `gradient_accumulation_steps` |
| 输入分辨率太高 | 图像尺寸 | 缩小输入，或使用多尺度训练 |
| 模型太大 | 参数量 | 用更小模型或混合精度训练 |
| 推理时保留计算图 | `with torch.no_grad()` 缺失 | 推理和验证时加 `no_grad()` |
| 中间特征图未释放 | 存储了大量张量 | 不要在 list 中存储 batch 级别张量 |
| 多进程 DataLoader | `num_workers` 太大 | 减小 `num_workers`，或设置 `pin_memory=False` |

### 混合精度训练（节省约 50% 显存）

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with autocast():                          # 自动使用 FP16
        outputs = model(batch['image'])
        loss = criterion(outputs, batch['label'])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 梯度检查点（极度节省显存，适合大模型）

```python
from torch.utils.checkpoint import checkpoint

# 使用梯度检查点（以计算换显存，速度约慢 20%）
class MyModel(nn.Module):
    def forward(self, x):
        # 对显存瓶颈层使用 checkpoint
        x = checkpoint(self.expensive_layer, x)
        return x
```

💡 **Tips**：`torch.cuda.memory_summary()` 输出详细的分配信息，是定位 OOM 的最有效工具。建议在 OOM 发生后立即调用，查看哪个张量占了最多显存。
