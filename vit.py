import torch
import torch.nn as nn
from transformer import Encoder

from einops import rearrange
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PositionalEncoding(nn.Module):
    def __init__(self, h, w, dim) -> None:
        super().__init__()
        assert dim % 4 == 0
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        angle_rads = 1.0 / 1000 ** (torch.arange(dim // 4) / (dim // 4 - 1))
        y = y.flatten()[:, None] * angle_rads[None, :]
        x = x.flatten()[:, None] * angle_rads[None, :]
        self.pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

    def forward(self, x):
        return x + self.pe
    
class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, d_model, n_heads, hidden_size, MoE, n_blocks, channels=3,) -> None:
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        patch_dim = patch_height * patch_width * channels
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_embedding = PositionalEncoding(image_height // patch_height, image_width // patch_width, d_model)
        self.encoder = Encoder(d_model, n_heads, hidden_size, MoE, n_blocks)

        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(d_model, num_classes)

    def forward(self, img):
        x = self.patch_embedding(img)
        x = self.pos_embedding(x)

        x = self.encoder(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
    
img_size = 256
patch_size = 8
num_classes = 1000
d_model = 128
n_heads = 8
hidden_size = 512
MoE = 4
n_blocks = 6
    
model = ViT(img_size, patch_size, num_classes, d_model, n_heads, hidden_size, MoE, n_blocks)

x = torch.randn(size=(4, 3, 256, 256))
y = model(x)
print(y.shape)
