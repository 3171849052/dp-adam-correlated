"""Exact joint transformed norms with layer-at-a-time per-example gradients."""
import torch
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from exp3.geometry import freeze_geometry


class GeometryGhostModule(GradSampleModuleFastGradientClipping):
    def __init__(self, model, max_grad_norm):
        self.NORM_SAMPLERS = {}
        for layer_type, sampler in self.GRAD_SAMPLERS.items():
            def norm_sampler(layer, activations, backprops, sampler=sampler):
                samples = sampler(layer, activations, backprops.float())
                return {p: p._geometry.transform(gs.float()).flatten(1).norm(dim=1)
                        for p, gs in samples.items()}
            self.NORM_SAMPLERS[layer_type] = norm_sampler
        super().__init__(model, loss_reduction='sum', max_grad_norm=max_grad_norm,
                         use_ghost_clipping=True)


def clipped_microbatch(model, inputs, targets, clip):
    losses = torch.nn.functional.cross_entropy(model(inputs), targets, reduction='none')
    losses.sum().backward(retain_graph=True)
    norms = model.get_norm_sample().detach()
    factors = (clip / norms.clamp_min(1e-30)).clamp(max=1)
    model.zero_grad(set_to_none=True)
    model.disable_hooks()
    (losses * factors).sum().backward()
    model.enable_hooks()
    return float(losses.detach().sum()), int((norms > clip).sum())


class LogicalBatch:
    def __init__(self, model, optimizer, noise, method, clip, lambda_parallel=1.,
                 kappa=4., rho=.1, accumulation=4, logical_size=1000):
        self.model, self.optimizer, self.noise = model, optimizer, noise
        self.method, self.clip = method, clip
        self.lambda_parallel, self.kappa, self.rho = lambda_parallel, kappa, rho
        self.accumulation, self.logical_size = accumulation, logical_size
        self.parameters = list(model.parameters())
        self.sums = [torch.zeros_like(p) for p in self.parameters]
        self.micro_steps = self.optimizer_steps = 0
        self.geometries = None

    def begin_microbatch(self):
        if self.micro_steps == 0:
            self.geometries = freeze_geometry(self.optimizer, self.method,
                                              self.lambda_parallel, self.kappa, self.rho)
            for p in self.parameters:
                p._geometry = self.geometries[p]
            for total in self.sums:
                total.zero_()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def finish_microbatch(self):
        for total, p in zip(self.sums, self.parameters):
            total.add_(p.grad)
        self.micro_steps += 1
        if self.micro_steps != self.accumulation:
            return False
        # Linearity lets us transform the clipped sum after microbatch accumulation.
        # Exactly one noise.next() draws the joint full-model release.
        noises = self.noise.next() if self.noise is not None else None
        for i, (total, p) in enumerate(zip(self.sums, self.parameters)):
            if noises is not None:
                s = self.geometries[p]
                total = s.inverse(s.transform(total) + noises[i])
            p.grad = total / self.logical_size
        self.optimizer.step()
        self.optimizer_steps += 1
        self.micro_steps = 0
        return True
