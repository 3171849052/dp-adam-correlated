"""Exact pre-clipping norms and one update per logical batch."""
import numpy as np
import torch


def parameter_norms(model):
    parts = {"backbone_norm": [], "head_norm": []}
    for name, p in model._module.named_parameters():
        assert p.requires_grad
        parts["head_norm" if name.startswith("head.") else "backbone_norm"].append(p._norm_sample.detach().square())
    squares = {key: torch.stack(values).sum(0) for key, values in parts.items()}
    return dict(full_model_norm=(squares["backbone_norm"] + squares["head_norm"]).sqrt(),
                **{key: value.sqrt() for key, value in squares.items()})


def clipped_microbatch(model, inputs, targets, max_grad_norm):
    losses = torch.nn.functional.cross_entropy(model(inputs), targets, reduction="none")
    losses.sum().backward(retain_graph=True)
    norms = parameter_norms(model)
    coefficients = (max_grad_norm / norms["full_model_norm"].clamp_min(1e-30)).clamp(max=1.0)
    model.zero_grad(set_to_none=True)
    model.disable_hooks()
    (losses * coefficients).sum().backward()
    model.enable_hooks()
    return float(losses.detach().sum()), {key: value.cpu().numpy() for key, value in norms.items()}


class EpochNorms:
    # Retain only norms in RAM until the epoch ends; persist aggregates only.
    def __init__(self, clip):
        self.clip = clip
        self.values = {key: [] for key in ("full_model_norm", "backbone_norm", "head_norm")}

    def add(self, values):
        for key, value in values.items():
            self.values[key].append(value)

    def aggregate(self):
        result = {}
        for key, chunks in self.values.items():
            values = np.concatenate(chunks)
            stats = dict(zip(("p10", "p25", "p50", "p75", "p90", "p99"),
                             map(float, np.quantile(values, [.1, .25, .5, .75, .9, .99]))))
            stats.update(mean=float(values.mean()), clip_fraction=float((values > self.clip).mean()))
            result[key] = stats
        return result


class IIDNoise:
    def __init__(self, parameters, std, seed):
        self.parameters = list(parameters)
        self.std = std
        self.generator = torch.Generator(device=self.parameters[0].device).manual_seed(seed)
        self.step_count = 0

    def next(self):
        self.step_count += 1
        return [torch.randn(p.shape, dtype=torch.float32, device=p.device,
                            generator=self.generator) * self.std for p in self.parameters]


class LogicalBatch:
    """Exactly one noise draw and optimizer step per accumulation window."""
    def __init__(self, model, optimizer, accumulation, logical_size, noise=None):
        self.model = model
        self.parameters = list(model.parameters())
        self.optimizer = optimizer
        self.accumulation = accumulation
        self.logical_size = logical_size
        self.noise = noise
        self.micro_steps = 0
        self.optimizer_steps = 0
        self.sums = [torch.zeros_like(p, dtype=torch.float32) for p in self.parameters]

    def begin_microbatch(self):
        if self.micro_steps == 0:
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
            if noise is not None:
                total.add_(noise[i])
            p.grad = total / self.logical_size
        self.optimizer.step()
        self.optimizer_steps += 1
        self.micro_steps = 0
        return True
