import torch
from pathlib import Path

# ⚠️ 修改这里：换成你数据集里的任意一个 .pt 文件的绝对路径
file_path = r"D:\000college\1cs\3projects\CycleGAN\datasets\monet2photo_pt\trainA\00151.pt"
# (注意：如果不知道具体文件名，去文件夹里随便复制一个名字填进去)

# 如果找不到文件，就自动找一个
if "some_image.pt" in file_path:
    data_root = Path(r"D:\000college\1cs\3projects\CycleGAN\datasets\monet2photo_pt")
    # 递归找第一个 .pt 文件
    file_path = next(data_root.rglob("*.pt"))
    print(f"自动定位到文件: {file_path}")

print(f"\n🔍 正在检查: {file_path}")
data = torch.load(file_path, map_location='cpu')

# 你的 .pt 可能存的是个字典，也可能直接是 Tensor
if isinstance(data, dict):
    print(f"数据结构: Dict (包含 keys: {list(data.keys())})")
    # 假设核心数据存在 'content' 或 'data' 键里，根据你之前的代码逻辑猜测是 'content'
    tensor = data.get('content', list(data.values())[0]) 
else:
    print("数据结构: Tensor")
    tensor = data

if not torch.is_tensor(tensor):
    print("❌ 错误: 文件里存的不是 Tensor，无法分析数值。")
else:
    print("-" * 30)
    print(f"📏 形状 (Shape): {tensor.shape}")
    print(f"🔢 数据类型 (Dtype): {tensor.dtype}")
    print(f"📉 最小值 (Min):   {tensor.min().item():.4f}")
    print(f"📈 最大值 (Max):   {tensor.max().item():.4f}")
    print(f"📊 平均值 (Mean):  {tensor.mean().item():.4f}")
    print(f"📉 标准差 (Std):   {tensor.std().item():.4f}")
    print("-" * 30)

    # === 自动判决 ===
    if tensor.max() > 10.0:
        print("🚨 结论: 这是【原始像素 (Pixel)】！")
        print("   原因: 最大值超过 10，通常是 0-255 的 RGB 值。")
        print("   后果: 必须在 Dataset 里除以 255 并归一化，或者重新预处理数据。")
    elif tensor.shape[0] == 3 and tensor.shape[1] > 64:
        print("🚨 结论: 这是【归一化后的像素】，但【未编码 (Not Latent)】！")
        print("   原因: 通道数是 3 (RGB)，且尺寸很大 (如 512x512)。")
        print("   后果: 需要经过 VAE Encoder 才能用于 Latent 训练。")
    elif tensor.shape[0] == 4 and tensor.shape[1] <= 128:
        print("✅ 结论: 这是【VAE Latent】！")
        print("   原因: 通道数是 4，尺寸较小 (如 64x64)，数值范围正常。")
    else:
        print("❓ 结论: 未知格式，请人工核对。")