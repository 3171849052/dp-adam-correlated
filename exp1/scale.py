"""Exact scaled norms in Opacus' two-backward Fast/Ghost clipping structure."""
import torch
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)


class ScaledGhostModule(GradSampleModuleFastGradientClipping):
    """Replace norm samplers by exact layerwise scaled Fast Clipping samplers.

    An arbitrary elementwise Adam scale does not factor into the usual
    Linear Ghost identity. Materialize only one layer's per-example gradients,
    multiply by its frozen FP32 scale, reduce immediately, and discard them.
    Opacus still handles activation/backprop hooks and the two-pass protocol.
    No full-model per-example gradients are retained.
    """
    def __init__(self, model, max_grad_norm):
        self.NORM_SAMPLERS = {}
        for layer_type, grad_sampler in self.GRAD_SAMPLERS.items():
            def scaled_norm(layer, activations, backprops, sampler=grad_sampler):
                samples = sampler(layer, activations, backprops.float())
                return {p: (gs.float() * p._logical_scale).flatten(1).norm(2, dim=1)
                        for p, gs in samples.items()}
            self.NORM_SAMPLERS[layer_type] = scaled_norm
        super().__init__(model, loss_reduction='sum', max_grad_norm=max_grad_norm,
                         use_ghost_clipping=True)
        for p in self.parameters():
            p._logical_scale = torch.ones_like(p, dtype=torch.float32)


@torch.no_grad()
def freeze_adam_scale(optimizer, eps_scale):
    """Use vhat of the previous completed Adam step; vhat_0=0."""
    scales = []
    for group in optimizer.param_groups:
        beta2 = group['betas'][1]
        for p in group['params']:
            state = optimizer.state[p]
            if len(state) == 0:
                vhat = torch.zeros_like(p, dtype=torch.float32)
            else:
                step = int(state['step'].item())
                vhat = state['exp_avg_sq'].float() / (1 - beta2 ** step)
            scale = 1.0 / (vhat.sqrt() + eps_scale)
            p._logical_scale = scale
            scales.append(scale)
    return scales


def clipped_microbatch(model, inputs, targets, max_grad_norm):
    """First backward measures norms; second backward sums clipped gradients."""
    losses = torch.nn.functional.cross_entropy(model(inputs), targets, reduction='none')
    losses.sum().backward(retain_graph=True)
    norms = model.get_norm_sample().detach()
    coefficients = (max_grad_norm / norms.clamp_min(1e-30)).clamp(max=1.0)
    model.zero_grad(set_to_none=True)
    model.disable_hooks()
    (losses * coefficients).sum().backward()
    model.enable_hooks()
    return float(losses.detach().sum()), int((norms > max_grad_norm).sum())


class LogicalBatch:
    """Exactly one noise draw and optimizer step per accumulation window."""
    def __init__(self, model, optimizer, accumulation, logical_size, noise=None,
                 scaled=False, eps_scale=0.001):
        self.model = model
        self.parameters = list(model.parameters())
        self.optimizer = optimizer
        self.accumulation = accumulation
        self.logical_size = logical_size
        self.noise = noise
        self.scaled = scaled
        self.eps_scale = eps_scale
        self.micro_steps = 0
        self.optimizer_steps = 0
        self.sums = [torch.zeros_like(p, dtype=torch.float32) for p in self.parameters]

    def begin_microbatch(self):
        if self.micro_steps == 0:
            if self.scaled:
                freeze_adam_scale(self.optimizer, self.eps_scale)
            for total in self.sums:
                total.zero_()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def finish_microbatch(self):
        for total, p in zip(self.sums, self.parameters):
            total.add_(p.grad.float())
        self.micro_steps += 1
        if self.micro_steps < self.accumulation:
            return False
        assert self.micro_steps == self.accumulation
        noise = self.noise.next() if self.noise is not None else None
        for i, (total, p) in enumerate(zip(self.sums, self.parameters)):
            if self.scaled:
                total.mul_(p._logical_scale)
            if noise is not None:
                total.add_(noise[i])
            if self.scaled:
                total.div_(p._logical_scale)
            p.grad = total / self.logical_size
        self.optimizer.step()
        self.optimizer_steps += 1
        self.micro_steps = 0
        return True
