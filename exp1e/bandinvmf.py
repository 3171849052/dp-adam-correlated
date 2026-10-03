"""JAX coefficient construction; PyTorch FP32 finite impulse response noise."""
import os

# JAX does coefficient work on CPU; it must not reserve a training GPU.
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'true'

import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platforms', 'cpu')
from jax_privacy.matrix_factorization import toeplitz
import numpy as np
import torch


def build_matrices(method, total_steps, num_bands, beta1):
    if method in ('adam', 'dp_adam'):
        noising_coef = np.array([1.0])
        workload = np.ones(total_steps)
    else:
        workload = jnp.ones(total_steps)
        if method == 'dp_adam_bandinvmf_momentum':
            decay = beta1 ** jnp.arange(total_steps)
            workload = toeplitz.multiply(jnp.ones(total_steps), decay, n=total_steps)
        noising_coef = np.asarray(toeplitz.banded_inverse_square_root_noising_coefs(
            num_bands, workload_coef=workload), dtype=np.float64)
    strategy_coef = toeplitz.inverse_coef(jnp.asarray(noising_coef), total_steps)
    strategy = np.asarray(toeplitz.materialize_lower_triangular(
        strategy_coef, n=total_steps), dtype=np.float64)
    return noising_coef, strategy, np.asarray(workload)


class BandInvMFNoise:
    """At logical step t return sum_l d[l] z[t-l], where D=C^{-1}.

    The retained history contains IID innovations, not previous outputs.
    num_bands=1 with d=[1] is exactly IID noise.
    """
    def __init__(self, parameters, coefficients, innovation_std, total_steps, seed):
        self.parameters = list(parameters)
        self.coefficients = np.asarray(coefficients)
        self.innovation_std = float(innovation_std)
        self.total_steps = total_steps
        self.step_count = 0
        self.history = []
        self.generator = torch.Generator(device=self.parameters[0].device)
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def next(self):
        assert self.step_count < self.total_steps, 'BandInvMF exhausted'
        innovations = [torch.randn(p.shape, device=p.device, dtype=torch.float32,
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
        return self.innovation_std * float(np.linalg.norm(
            self.coefficients[:min(step + 1, len(self.coefficients))]))
