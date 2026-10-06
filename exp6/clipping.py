"""Materialize per-example gradients only for the 98 trainable tensors."""
import torch
from torch.func import functional_call, grad_and_value, vmap
from torch.nn import functional as F


def per_example(model, inputs, targets):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}

    def objective(trainable, image, label):
        logits = functional_call(model, trainable, (image.unsqueeze(0),), strict=False)
        loss = F.cross_entropy(logits, label.unsqueeze(0))
        return loss, (logits.argmax(-1).squeeze(0) == label).float()

    gradients, (losses, correct) = vmap(grad_and_value(objective, has_aux=True),
                                     in_dims=(None, 0, 0))(params, inputs, targets)
    return {n: g.detach() for n, g in gradients.items()}, losses.detach(), correct.detach()


def global_norm(gradients, per_sample=True):
    if per_sample:
        return sum(g.flatten(1).square().sum(1) for g in gradients.values()).sqrt()
    return sum(g.square().sum() for g in gradients.values()).sqrt()


def clipped_sum(gradients, C):
    norms = global_norm(gradients)
    factors = torch.ones_like(norms) if C is None else (C / norms.clamp_min(1e-30)).clamp(max=1.)
    sums = {n: (g * factors.reshape((-1,) + (1,) * (g.ndim - 1))).sum(0)
            for n, g in gradients.items()}
    return sums, norms, factors
