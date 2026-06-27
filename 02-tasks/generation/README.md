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
