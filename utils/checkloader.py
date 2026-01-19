import torch
from torch.utils.data import DataLoader
from dataset import Stage1Dataset  # 确保引用你修改后的 Dataset

# 初始化你的 Dataset
ds = Stage1Dataset("D:\\000college\\1cs\\3projects\\CycleGAN\\datasets\\monet2photo_pt", num_classes=2)
dl = DataLoader(ds, batch_size=16, shuffle=True)

# 抓取一个 Batch
x_c, x_s, _, _ = next(iter(dl))

print("="*30)
print(f"🧐 正在检查送入 GPU 前的数据范围...")
print(f"Max:  {x_c.max().item():.4f}")
print(f"Min:  {x_c.min().item():.4f}")
print(f"Mean: {x_c.mean().item():.4f}")
print(f"Std:  {x_c.std().item():.4f}")
print("="*30)

# 判决标准：
# 1. 如果 Max 在 1.5 ~ 3.0 之间 -> ✅ 正常 (Scaled)。
# 2. 如果 Max 在 8.0 ~ 15.0 之间 -> ❌ 没缩放 (Raw)，会出红黑图。
# 3. 如果 Max 在 0.2 ~ 0.5 之间 -> ❌ 双重缩放 (Double Scaled)，会出灰图。