"""The identical 48 hidden matrix parameters for every nonlinear comparison."""
import torch


def expected_names():
    return tuple(f'blocks.{block}.{suffix}.weight' for block in range(12)
                 for suffix in ('attn.qkv', 'attn.proj', 'mlp.0', 'mlp.2'))


def support(model):
    parameters = dict(model.named_parameters())
    names = expected_names()
    assert len(names) == 48 and len(set(names)) == 48
    selected = {name: parameters[name] for name in names}
    assert all(p.requires_grad and p.ndim == 2 and p.dtype == torch.float32
               for p in selected.values())
    return selected
