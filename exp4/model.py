"""Local pretrained ViT-Tiny, with the same backbone and head as exp2."""
import hashlib
import os
from pathlib import Path

from exp4 import runtime
import timm
from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


class StableLayerNorm(nn.LayerNorm):
    """Same LayerNorm equation/parameters; fixed FP64 internal arithmetic.

    Large prescribed UC noise can overflow native FP32 backward reductions.
    Keep parameters and surrounding activations FP32; do not change eps.
    """
    def forward(self, x):
        return F.layer_norm(x.double(), self.normalized_shape,
                            self.weight.double(), self.bias.double(), self.eps).to(x.dtype)


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
        # Fixed FP64 math attention avoids overflow/cancellation in softmax
        # backward at the large logits induced by the prescribed noise grid.
        # Cast back before the projection; model parameters remain FP32.
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q.double(), k.double(), v.double(),
                                               dropout_p=0.0).to(x.dtype)
        return self.proj(y.transpose(1, 2).reshape(n, length, dim))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.norm1 = StableLayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = StableLayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTTiny(nn.Module):
    def __init__(self, patch_size=16, embed_dim=192, depth=12, heads=3,
                 mlp_ratio=4, num_classes=100, image_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.tokens = (image_size // patch_size) ** 2 + 1
        self.patch = nn.Linear(3 * patch_size ** 2, embed_dim)
        self.cls = nn.Embedding(1, embed_dim)
        self.position = nn.Embedding(self.tokens, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, heads, mlp_ratio)
                                      for _ in range(depth)])
        self.norm = StableLayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, images):
        n = images.shape[0]
        patches = F.unfold(images, self.patch_size, stride=self.patch_size).transpose(1, 2)
        x = self.patch(patches)
        cls = self.cls(torch.zeros(n, 1, device=images.device, dtype=torch.long))
        positions = torch.arange(self.tokens, device=images.device).repeat(n, 1)
        x = torch.cat((cls, x), dim=1) + self.position(positions)
        if self.training and torch.is_grad_enabled():
            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=False)
        else:
            x = self.blocks(x)
        return self.head(self.norm(x)[:, 0])

    def load_backbone(self, reference):
        """Map every timm backbone tensor exactly; retain the random 100-way head."""
        mapped = {}
        for name, tensor in reference.state_dict().items():
            if name.startswith('head.'):
                continue
            if name == 'cls_token':
                name, tensor = 'cls.weight', tensor.reshape(1, -1)
            elif name == 'pos_embed':
                name, tensor = 'position.weight', tensor.squeeze(0)
            elif name == 'patch_embed.proj.weight':
                name, tensor = 'patch.weight', tensor.flatten(1)
            elif name == 'patch_embed.proj.bias':
                name = 'patch.bias'
            name = name.replace('.mlp.fc1.', '.mlp.0.').replace('.mlp.fc2.', '.mlp.2.')
            mapped[name] = tensor
        mapped.update({name: value for name, value in self.state_dict().items()
                       if name.startswith('head.')})
        self.load_state_dict(mapped, strict=True)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_CACHE = ROOT / 'cache/huggingface/hub/models--timm--vit_tiny_patch16_224.augreg_in21k_ft_in1k'


def checkpoint_path():
    revision = (CHECKPOINT_CACHE / 'refs/main').read_text().strip()
    path = CHECKPOINT_CACHE / 'snapshots' / revision / 'model.safetensors'
    # Opening the file fails directly if either the snapshot or its blob is absent.
    assert path.resolve().is_relative_to(ROOT / 'cache')
    with path.open('rb'):
        pass
    return path


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pretrained_vit():
    """Strict backbone mapping; every pretrained tensor comes from local cache/."""
    path = checkpoint_path()
    reference = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    reference.load_state_dict(load_file(str(path), device='cpu'), strict=True)
    model = ViTTiny()
    model.load_backbone(reference)
    cfg = reference.pretrained_cfg
    metadata = {key: cfg[key] for key in ('mean', 'std', 'crop_pct', 'interpolation', 'hf_hub_id')}
    metadata.update(checkpoint_path=str(path.relative_to(ROOT)),
                    checkpoint_sha256=checkpoint_sha256(path))
    return model, metadata


def initialization_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
