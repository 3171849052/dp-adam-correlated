"""True per-example gradients, coordinate transform, global clip, DP SGDM."""
from exp5 import runtime
import torch
from torch.func import functional_call, grad_and_value, vmap
from exp2.bandinvmf import BandInvMFNoise
from exp5.config import CHUNK_SIZE


def tensor_norm(values):
    return float(torch.stack([v.square().sum() for v in values]).sum().sqrt())


def coordinate_transform(g, tau):
    return g / (g.abs() + tau)


class PerExample:
    def __init__(self, model, chunk_size=CHUNK_SIZE):
        self.model, self.chunk_size = model, chunk_size
        def loss(params, buffers, x, y):
            logits = functional_call(model, (params, buffers), (x.unsqueeze(0),))
            return torch.nn.functional.cross_entropy(logits, y.unsqueeze(0)), logits.squeeze(0)
        self.batched = vmap(grad_and_value(loss, has_aux=True), in_dims=(None, None, 0, 0),
                            randomness='error')

    def chunks(self, inputs, targets):
        params, buffers = dict(self.model.named_parameters()), dict(self.model.named_buffers())
        for start in range(0, len(targets), self.chunk_size):
            with torch.no_grad():
                gradients, (losses, logits) = self.batched(
                    params, buffers, inputs[start:start+self.chunk_size], targets[start:start+self.chunk_size])
            yield gradients, losses, logits
            del gradients, losses, logits


@torch.no_grad()
def transform_clip(gradients, tau, C):
    """Sum norms across ALL parameter tensors before applying one sample factor."""
    n = next(iter(gradients.values())).shape[0]
    raw_norms = torch.stack([g.reshape(n, -1).square().sum(1) for g in gradients.values()]).sum(0).sqrt()
    for name, g in gradients.items():
        gradients[name] = coordinate_transform(g, tau)
    h_norms = torch.stack([h.reshape(n, -1).square().sum(1) for h in gradients.values()]).sum(0).sqrt()
    factors = (C / h_norms).clamp(max=1.)
    for h in gradients.values():
        h.mul_(factors.reshape((n,) + (1,) * (h.ndim - 1)))
    return raw_norms, h_norms, factors


@torch.no_grad()
def clip_microbatch(per_example, inputs, targets, tau, C=1.):
    params = list(per_example.model.parameters())
    sums = [torch.zeros_like(p) for p in params]
    raw_sums = [torch.zeros_like(p) for p in params]
    stats = dict(loss_sum=0., correct=0, count=len(targets), clipped=0,
                 raw_norm_sum=0., h_norm_sum=0., h_norm_sq_sum=0., factor_sum=0., q_norm_sum=0.)
    offset = 0
    for gradients, losses, logits in per_example.chunks(inputs, targets):
        for total, g in zip(raw_sums, gradients.values()):
            total.add_(g.sum(0))
        raw, h, factors = transform_clip(gradients, tau, C)
        for total, q in zip(sums, gradients.values()):
            total.add_(q.sum(0))
        stats['loss_sum'] += float(losses.sum())
        stats['correct'] += int((logits.argmax(1) == targets[offset:offset+len(losses)]).sum())
        offset += len(losses)
        stats['raw_norm_sum'] += float(raw.sum())
        stats['h_norm_sum'] += float(h.sum())
        stats['h_norm_sq_sum'] += float(h.square().sum())
        stats['factor_sum'] += float(factors.sum())
        stats['clipped'] += int((factors < 1).sum())
        stats['q_norm_sum'] += float((h * factors).sum())
        del gradients, losses, logits, raw, h, factors
    for p, total in zip(params, sums):
        p.grad = total
    return stats, raw_sums


class LogicalSGDM:
    def __init__(self, parameters, noise, lr, logical_size=1000, physical_size=100, beta=.9):
        self.parameters = list(parameters)
        self.noise, self.lr, self.beta = noise, lr, beta
        self.logical_size, self.physical_size = logical_size, physical_size
        if logical_size % physical_size:
            raise ValueError('Logical size must be divisible by physical size')
        self.accumulation = logical_size // physical_size
        self.sums = [torch.zeros_like(p) for p in self.parameters]
        self.raw_sums = [torch.zeros_like(p) for p in self.parameters]
        self.momentum = [torch.zeros_like(p) for p in self.parameters]
        self.micro_steps = self.steps = 0
        self.q_norm_sum = 0.

    @torch.no_grad()
    def finish_microbatch(self, raw_sums, q_norm_sum, count=100):
        if count != self.physical_size:
            raise ValueError(f'Expected physical batch {self.physical_size}, got {count}')
        for total, raw_total, p, raw in zip(self.sums, self.raw_sums, self.parameters, raw_sums):
            total.add_(p.grad)
            raw_total.add_(raw)
        self.q_norm_sum += q_norm_sum
        self.micro_steps += 1
        if self.micro_steps < self.accumulation:
            return None
        assert self.micro_steps == self.accumulation
        average = [s / self.logical_size for s in self.sums]
        query_norm = tensor_norm(average)
        raw_mean_gradient_norm = tensor_norm(self.raw_sums) / self.logical_size
        coherence = query_norm * self.logical_size / self.q_norm_sum if self.q_norm_sum else 0.
        noise = self.noise.next()
        for p, m, q, n in zip(self.parameters, self.momentum, average, noise):
            m.mul_(self.beta).add_(q + n)
            p.add_(m, alpha=-self.lr)
        momentum_norm = tensor_norm(self.momentum)
        metrics = dict(query_norm=query_norm, raw_mean_gradient_norm=raw_mean_gradient_norm,
                       batch_coherence=coherence, noise_std=self.noise.marginal_std(self.steps),
                       momentum_norm=momentum_norm, update_norm=self.lr * momentum_norm)
        for total in self.sums + self.raw_sums:
            total.zero_()
        self.micro_steps = 0
        self.q_norm_sum = 0.
        self.steps += 1
        return metrics
