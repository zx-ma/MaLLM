import torch

from model import Attention

attn = Attention(4)
x = torch.randn(2, 3, 4)
out = attn(x)
print(out.shape)
