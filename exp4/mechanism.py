"""Both methods use the same innovation filter; IID is D=I."""
import numpy as np
import torch


class ZeroNoise:
    """Stage 0 only: no Gaussian draws and no privacy claim."""
    def __init__(self, parameters, total_steps):
        self.parameters = list(parameters)
        self.total_steps = total_steps
        self.step_count = 0

    def next(self):
        assert self.step_count < self.total_steps
        self.step_count += 1
        return [torch.zeros_like(p) for p in self.parameters]

    def marginal_std(self, step):
        return 0.0


class TemporalGaussianNoise:
    def __init__(self, parameters, coefficients, innovation_std, total_steps, seed):
        self.parameters = list(parameters)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.innovation_std = float(innovation_std)
        self.total_steps = total_steps
        self.step_count = 0
        self.history = []
        self.generator = torch.Generator(device=self.parameters[0].device)
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def next(self):
        assert self.step_count < self.total_steps
        innovations = [torch.randn(p.shape, device=p.device, dtype=p.dtype,
                                   generator=self.generator) * self.innovation_std
                       for p in self.parameters]
        current = [z * float(self.coefficients[0]) for z in innovations]
        for lag, previous in enumerate(self.history, start=1):
            for output, z in zip(current, previous):
                output.add_(z, alpha=float(self.coefficients[lag]))
        self.history.insert(0, innovations)
        del self.history[len(self.coefficients) - 1:]
        self.step_count += 1
        return current

    def marginal_std(self, step):
        assert 0 <= step < self.total_steps
        return self.innovation_std * float(np.linalg.norm(
            self.coefficients[:min(step + 1, len(self.coefficients))]))
