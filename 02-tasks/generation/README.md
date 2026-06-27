# 图像生成

## 主要范式

### GAN（生成对抗网络）

| 模型 | 特点 |
|------|------|
| StyleGAN2/3 | 高质量人脸，隐空间可控 |
| BigGAN | 高分辨率，类别条件 |
| Pix2Pix | 图像翻译，配对数据 |
| CycleGAN | 无配对图像翻译 |

### Diffusion Model（扩散模型，当前主流）

| 模型 | 特点 |
|------|------|
| DDPM | 扩散模型基础 |
| DDIM | 加速采样 |
| Stable Diffusion | 潜空间扩散，开源最流行 |
| ControlNet | 条件控制（边缘/姿态/深度）|
| IP-Adapter | 图像提示适配 |

### 评估指标

| 指标 | 说明 |
|------|------|
| FID | Fréchet Inception Distance，越低越好 |
| IS | Inception Score，越高越好 |
| LPIPS | 感知相似度 |
| CLIP Score | 文图对齐度 |

## 应用方向

- 文生图（Text-to-Image）
- 图像编辑（Inpainting / Style Transfer）
- 数据增强（合成训练数据）
- 视频生成（Sora / CogVideo）

---

## DDPM：前向加噪与反向去噪

### 前向过程（加噪）

```python
import torch
import torch.nn.functional as F
import numpy as np

class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.T = T
        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1.0 - self.betas
        # alphas_cumprod: α̅_t = ∏_{i=1}^{t} α_i
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """
        前向扩散：x_0 → x_t（一步直达任意时刻）
        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    def p_losses(self, model, x_0, t):
        """训练损失：预测在时刻 t 加入的噪声"""
        x_noisy, noise = self.q_sample(x_0, t)
        noise_pred = model(x_noisy, t)
        return F.mse_loss(noise_pred, noise)
```

### 反向过程（去噪采样）

```python
    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        单步去噪：x_t → x_{t-1}
        x_{t-1} = 1/sqrt(α_t) * (x_t - β_t/sqrt(1-α̅_t) * ε_θ(x_t, t)) + σ_t * z
        """
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]

        # 网络预测噪声
        noise_pred = model(x_t, t)

        # 计算均值
        coef = beta_t / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - coef * noise_pred)

        # 加噪（t > 0 时）
        if t > 0:
            variance = beta_t
            z = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * z
        return mean

    @torch.no_grad()
    def sample(self, model, shape):
        """从纯噪声 x_T 逐步去噪到 x_0"""
        x = torch.randn(shape).to(next(model.parameters()).device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, dtype=torch.long,
                                  device=x.device)
            x = self.p_sample(model, x, t)
        return x.clamp(-1, 1)
```

---

## Stable Diffusion 推理代码

Stable Diffusion 在**潜空间**（Latent Space）中进行扩散，分辨率效率是像素空间的 8 倍。

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# 加载模型（首次会下载，约 4GB）
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,       # 关闭安全检查（仅限研究用途）
)
# 换用更快的调度器（20 步即可）
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()  # 显存优化

# 文生图
image = pipe(
    prompt="a photo of an astronaut riding a horse on mars, 8k, photorealistic",
    negative_prompt="blurry, ugly, low quality",
    num_inference_steps=20,
    guidance_scale=7.5,   # CFG scale：越高越忠实 prompt，越低越自由
    width=512, height=512,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("output.png")
```

---

## ControlNet：条件控制机制

ControlNet 通过添加与 UNet 并行的**条件编码器**，实现边缘图/姿态/深度等精确控制：

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch, cv2, numpy as np

# 加载 ControlNet（Canny 边缘控制）
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

# 1. 提取控制信号（Canny 边缘）
image = load_image("input.jpg")
image_np = np.array(image)
canny = cv2.Canny(image_np, 100, 200)
canny_rgb = np.stack([canny] * 3, axis=-1)
from PIL import Image
control_image = Image.fromarray(canny_rgb)

# 2. 生成（保持结构，改变风格/内容）
result = pipe(
    prompt="a high quality photo of a building, golden hour",
    image=control_image,
    num_inference_steps=20,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,  # 控制强度（0~2）
).images[0]
result.save("controlled_output.png")
```

---

## 图像编辑技术

### SDEdit（半扩散重绘）

```python
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

result = pipe(
    prompt="a beautiful sunset over the ocean",
    image=load_image("source.jpg"),
    strength=0.75,  # 噪声强度：越高改动越大（0~1）
    guidance_scale=7.5,
    num_inference_steps=50,
).images[0]
```

### Inpainting（局部重绘）

```python
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
).to("cuda")

# mask 中白色区域将被重绘
result = pipe(
    prompt="a beautiful garden with flowers",
    image=load_image("original.jpg"),
    mask_image=load_image("mask.png"),   # 白色=重绘区域
    num_inference_steps=50,
).images[0]
```

---

## GAN 训练技巧与常见失败模式

### 训练技巧

```python
# 1. 标签平滑（防止判别器过于自信）
real_label = torch.ones(B) * 0.9   # 而非精确的 1.0
fake_label = torch.zeros(B) + 0.1  # 而非精确的 0.0

# 2. 梯度惩罚（WGAN-GP，稳定训练）
def gradient_penalty(discriminator, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interp, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True
    )[0]
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# 3. 频率比（每训练 G 一次，训练 D n_critic 次）
for _ in range(n_critic):
    train_discriminator(real_batch)
train_generator()
```

### 常见失败模式

| 现象 | 原因 | 解决方案 |
|------|------|---------|
| **模式崩塌** | G 只生成少数几种输出 | 梯度惩罚、MiniBatch Discrimination |
| **训练不稳定** | 损失震荡、NaN | 降低学习率、使用 WGAN-GP |
| **棋盘格伪影** | 转置卷积步长不整除 | 改用上采样 + 普通卷积 |
| **训练不同步** | D 太强/太弱 | 调整 n_critic 或学习率比 |

---

## FID 计算代码

FID（Fréchet Inception Distance）衡量真实图像分布与生成图像分布的差异：

```python
import torch
import numpy as np
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

def compute_fid(real_images, fake_images, batch_size=50, device='cuda'):
    """
    real_images: [N, 3, H, W]，归一化到 [-1, 1]
    fake_images: [N, 3, H, W]
    """
    # 加载 Inception-v3（使用 pool3 特征，2048 维）
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()  # 去掉分类头
    model.eval()

    def get_features(images):
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            # Inception 需要 299×299 输入
            batch = torch.nn.functional.interpolate(
                batch, size=(299, 299), mode='bilinear', align_corners=False
            )
            with torch.no_grad():
                feat = model(batch)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)

    real_feat = get_features(real_images)  # [N, 2048]
    fake_feat = get_features(fake_images)

    # 计算均值和协方差
    mu_r, sigma_r = real_feat.mean(0), np.cov(real_feat, rowvar=False)
    mu_f, sigma_f = fake_feat.mean(0), np.cov(fake_feat, rowvar=False)

    # FID = ||mu_r - mu_f||^2 + Tr(Σ_r + Σ_f - 2*sqrt(Σ_r*Σ_f))
    diff = mu_r - mu_f
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # 数值误差导致微小虚部，取实部
    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)
```

**踩坑记录**：
- FID 需要至少 **10000 张**真实图像和生成图像才能得到稳定估计，样本太少结果方差很大
- DDPM 原版 1000 步推理很慢，生产/评估时用 DDIM 20~50 步采样即可
- ControlNet conditioning scale > 1.5 时可能导致 artifact，建议从 1.0 开始调参
