"""One exact Ghost clipping implementation; DP average precedes momentum."""
import torch
from exp2.bandinvmf import BandInvMFNoise
from exp2.scale import clipped_microbatch
from exp5.config import FIXED


def tensor_norm(values):
    return float(torch.stack([v.square().sum() for v in values]).sum().sqrt())


def clip_microbatch(model, inputs, targets, C):
    # Reuse exp2's validated two-backward global clipping, without scaled geometry.
    correct = []
    hook = model.register_forward_hook(
        lambda module, args, output: correct.append(int((output.detach().argmax(1) == targets).sum())))
    loss, clipped = clipped_microbatch(model, inputs, targets, C)
    hook.remove()
    norms = model.get_norm_sample().detach()
    factors = (C / norms.clamp_min(1e-30)).clamp(max=1.)
    return dict(loss_sum=loss, clipped=clipped, norm_sum=float(norms.sum()),
                factor_sum=float(factors.sum()), count=targets.numel(), correct=correct[0])


class LogicalSGDM:
    def __init__(self, parameters, noise, lr, logical_size=1000, physical_size=100, beta=.9):
        self.parameters = list(parameters)
        self.noise, self.lr, self.beta = noise, lr, beta
        self.logical_size, self.physical_size = logical_size, physical_size
        if logical_size % physical_size:
            raise ValueError('Logical size must be divisible by physical size')
        self.accumulation = logical_size // physical_size
        self.sums = [torch.zeros_like(p) for p in self.parameters]
        self.momentum = [torch.zeros_like(p) for p in self.parameters]
        self.micro_steps = self.steps = 0

    @torch.no_grad()
    def finish_microbatch(self):
        for total, p in zip(self.sums, self.parameters):
            total.add_(p.grad)
        self.micro_steps += 1
        if self.micro_steps < self.accumulation:
            return None
        assert self.micro_steps == self.accumulation
        average = [s / self.logical_size for s in self.sums]
        query_norm = tensor_norm(average)
        noise = self.noise.next()  # Exactly once, after all physical batches.
        noisy = [q + n for q, n in zip(average, noise)]
        for p, m, dp in zip(self.parameters, self.momentum, noisy):
            m.mul_(self.beta).add_(dp)
            p.add_(m, alpha=-self.lr)
        momentum_norm = tensor_norm(self.momentum)
        metrics = dict(query_norm=query_norm, noise_std=self.noise.marginal_std(self.steps),
                       momentum_norm=momentum_norm, update_norm=self.lr * momentum_norm)
        for total in self.sums:
            total.zero_()
        self.micro_steps = 0
        self.steps += 1
        return metrics
