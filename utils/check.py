import torch

# 加载任意一个 .pt 文件
data = torch.load(r"D:\000college\1cs\3projects\CycleGAN\datasets\monet2photo_pt\trainA\00001.pt") 
# data['content'] 可能是 [4, 32, 32] 或者 [4, 64, 64]

print(data.shape)