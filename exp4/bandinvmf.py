"""BandInvMF Toeplitz coefficients, copied and narrowed from exp2.

The only workload is SGD/prefix sum. D=C^{-1} is a finite impulse response
filter of length num_bands; C is its full lower triangular inverse.
"""
from exp4 import runtime
import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import toeplitz
import numpy as np

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platforms', 'cpu')


def materialize(coefficients, total_steps):
    return np.asarray(toeplitz.materialize_lower_triangular(
        jnp.asarray(coefficients), n=total_steps), dtype=np.float64)


def build_matrices(total_steps, num_bands=4, identity=False):
    assert total_steps >= num_bands >= 1
    workload = np.ones(total_steps, dtype=np.float64)
    coefficients = np.array([1.0]) if identity else np.asarray(
        toeplitz.banded_inverse_square_root_noising_coefs(
            num_bands, workload_coef=jnp.asarray(workload)), dtype=np.float64)
    inverse = toeplitz.inverse_coef(jnp.asarray(coefficients), total_steps)
    return coefficients, materialize(inverse, total_steps), workload
