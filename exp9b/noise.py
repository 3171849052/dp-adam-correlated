"""FP32 IIR noise; the retained history contains previous outputs."""
import numpy as np
import torch
from scipy.linalg import toeplitz
from scipy.signal import lfilter


def materialize(coefficients, total_steps):
    column = np.zeros(total_steps, dtype=np.float64)
    column[:len(coefficients)] = coefficients
    return toeplitz(column, np.r_[column[0], np.zeros(total_steps-1)])


def inverse_coefficients(coefficients, total_steps):
    impulse = np.zeros(total_steps); impulse[0] = 1.
    return lfilter([1.], coefficients, impulse)


class BandMFNoise:
    def __init__(self, parameters, coefficients, innovation_std, total_steps, seed):
        self.parameters = list(parameters)
        self.coefficients = np.asarray(coefficients)
        assert self.coefficients[0] == 1.
        self.innovation_std = float(innovation_std)
        self.total_steps = total_steps
        self.step_count = 0
        self.history = []
        self.impulse = inverse_coefficients(self.coefficients, total_steps)
        self.generator = torch.Generator(device=self.parameters[0].device)
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def next(self):
        assert self.step_count < self.total_steps, 'BandMF exhausted'
        current = [torch.randn(p.shape, device=p.device, dtype=torch.float32,
                               generator=self.generator) * self.innovation_std
                   for p in self.parameters]
        for lag, previous in enumerate(self.history, start=1):
            for output, eta in zip(current, previous):
                output.add_(eta, alpha=-float(self.coefficients[lag]))
        self.history.insert(0, current)
        del self.history[len(self.coefficients)-1:]
        self.step_count += 1
        return current

    def marginal_std(self, step):
        return self.innovation_std * float(np.linalg.norm(self.impulse[:step+1]))
