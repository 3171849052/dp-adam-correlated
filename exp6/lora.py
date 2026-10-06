"""Standard W_eff = W + (alpha/r) B A, without backbone gradients."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)

    def effective_weight(self):
        return self.base.weight + self.scale * (self.B @ self.A)


def inject(model, rank=8, alpha=8):
    model.requires_grad_(False)
    for block in model.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
        block.attn.proj = LoRALinear(block.attn.proj, rank, alpha)
        block.mlp[0] = LoRALinear(block.mlp[0], rank, alpha)
        block.mlp[2] = LoRALinear(block.mlp[2], rank, alpha)
    model.head.requires_grad_(True)
    layers = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    assert len(layers) == 48
    assert all(torch.equal(m.effective_weight(), m.base.weight) for m in layers.values())
    expected = {f'{name}.{p}' for name in layers for p in ('A', 'B')} | {'head.weight', 'head.bias'}
    assert {n for n, p in model.named_parameters() if p.requires_grad} == expected
    return layers
