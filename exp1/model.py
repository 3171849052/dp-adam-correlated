"""CIFAR-native ViT-Tiny, built from Opacus-supported parameterized layers."""
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        n, length, dim = x.shape
        qkv = self.qkv(x).reshape(n, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(y.transpose(1, 2).reshape(n, length, dim))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTTiny(nn.Module):
    def __init__(self, patch_size=4, embed_dim=192, depth=12, heads=3,
                 mlp_ratio=4, num_classes=100):
        super().__init__()
        self.patch_size = patch_size
        self.tokens = (32 // patch_size) ** 2 + 1
        self.patch = nn.Linear(3 * patch_size ** 2, embed_dim)
        self.cls = nn.Embedding(1, embed_dim)
        self.position = nn.Embedding(self.tokens, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, heads, mlp_ratio)
                                      for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, images):
        n = images.shape[0]
        patches = F.unfold(images, self.patch_size, stride=self.patch_size).transpose(1, 2)
        x = self.patch(patches)
        cls = self.cls(torch.zeros(n, 1, device=images.device, dtype=torch.long))
        positions = torch.arange(self.tokens, device=images.device).repeat(n, 1)
        x = torch.cat((cls, x), dim=1) + self.position(positions)
        return self.head(self.norm(self.blocks(x))[:, 0])
