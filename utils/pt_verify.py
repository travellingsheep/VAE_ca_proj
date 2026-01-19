import torch
from pathlib import Path
from diffusers import AutoencoderKL
from PIL import Image

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode(latents: torch.Tensor, vae: AutoencoderKL, vae_dtype: torch.dtype) -> Image.Image:
    """
    使用 VAE 将 latent 表示解码为图片。

    Args:
        latents (torch.Tensor): latent 表示。
        vae (AutoencoderKL): VAE 模型。
        vae_dtype (torch.dtype): VAE 的数据类型。

    Returns:
        Image.Image: 解码后的图片。
    """
    latents = latents.to(device)
    with torch.cuda.amp.autocast(enabled=(vae_dtype == torch.float16)):
        imgs = vae.decode(latents / 0.18215).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    arr = (imgs[0].detach().float().cpu().permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype('uint8')
    return Image.fromarray(arr)

def restore_pt_to_image(pt_file: str, output_dir: str) -> None:
    """
    将 .pt 文件中的 latent 表示还原为图片。

    Args:
        pt_file (str): .pt 文件路径。
        output_dir (str): 输出图片的保存目录。
    """
    # 加载 latent 表示
    latent = torch.load(pt_file)

    # 确保 latent 是正确的形状
    if not isinstance(latent, torch.Tensor):
        raise ValueError("加载的 .pt 文件内容不是 Tensor 类型")

    # 将 latent 转移到设备
    latent = latent.to(device)

    # 加载 VAE 模型
    print("Loading VAE...")
    vae_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device, dtype=vae_dtype)
    vae.eval()
    vae.requires_grad_(False)

    # 解码为图片
    img = decode(latent, vae, vae_dtype)

    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存图片
    img_name = Path(pt_file).stem + "_restored.jpg"
    img.save(output_path / img_name, quality=95)
    print(f"✅ 已还原图片: {output_path / img_name}")

def main():
    # 示例调用
    pt_file = r"D:\000college\1cs\3projects\CycleGAN\datasets\monet2photo_pt\trainB\2013-11-08 16_45_24.pt"  # 替换为你的 .pt 文件路径
    output_dir = "pt_verify_outputs"  # 替换为保存图片的目录路径
    restore_pt_to_image(pt_file, output_dir)

if __name__ == '__main__':
    main()