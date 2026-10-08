"""Exp2 filters plus full-matrix optimization for non-Toeplitz momentum bias."""
from functools import lru_cache
import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from exp2.bandinvmf import BandInvMFNoise, materialize, build_matrices as exp2_matrices
from exp2.privacy import fixed_epoch_sensitivity


def momentum_bias_workload(n, beta1):
    # One-indexed k,j: cumulative bias-corrected first moment, omitting (1-beta1).
    t = np.arange(1, n + 1)
    lag = t[:, None] - t[None, :]
    moments = np.where(lag >= 0, beta1 ** np.maximum(lag, 0), 0.)
    moments /= (1 - beta1 ** t)[:, None]
    return np.tril(np.cumsum(moments, axis=0))


def inverse_strategy(d, n):
    inverse = np.zeros(n)
    inverse[0] = 1 / d[0]
    for t in range(1, n):
        width = min(t, len(d) - 1)
        inverse[t] = -np.dot(d[1:width + 1], inverse[t - width:t][::-1]) / d[0]
    return toeplitz(inverse, np.r_[inverse[0], np.zeros(n - 1)])


def workload_error(W, d):
    """Mean squared row error of the actual complete W @ D matrix."""
    return float(np.sum((W @ materialize(d, len(W))) ** 2) / len(W))


def optimize_bias_filter(W, initial, k=5, spacing=50):
    n, bands = len(W), len(initial)
    assert n == k * spacing and W.shape == (n, n)
    # Q[a,b] = <W shift_a, W shift_b>; this uses every entry of W,
    # and exactly equals ||W D||_F^2, without treating W as Toeplitz.
    shifted = np.stack([np.pad(W[:, lag:], ((0, 0), (0, lag))) for lag in range(bands)])
    Q = np.einsum('aij,bij->ab', shifted, shifted) / n
    def objective(x):
        d = np.r_[1., x]  # overall filter scale cancels sensitivity^2 * error
        with np.errstate(over='ignore', invalid='ignore'):
            strategy = inverse_strategy(d, n)
            if not np.isfinite(strategy).all() or np.max(np.abs(strategy)) > 1e8:
                return 1e100
            # Same absolute Gram bound as Exp2, evaluated only for participating columns.
            indices = np.arange(spacing)[:, None] + spacing * np.arange(k)[None, :]
            columns = strategy[:, indices]
            grams = np.einsum('njk,njl->jkl', columns, columns)
            sensitivity_squared = np.max(np.abs(grams).sum(axis=(1, 2)))
            return float(sensitivity_squared * (d @ Q @ d))
    starts = (initial[1:] / initial[0], np.zeros(bands - 1))
    fits = [minimize(objective, x, method='Nelder-Mead',
                     options=dict(maxiter=1600, xatol=1e-9, fatol=1e-7)) for x in starts]
    fit = min(fits, key=lambda r: r.fun)
    assert fit.success, fit.message
    d = np.r_[1., fit.x]
    strategy = inverse_strategy(d, n)
    exact = fixed_epoch_sensitivity(strategy, k, spacing) ** 2 * workload_error(W, d)
    assert np.isclose(exact, fit.fun, rtol=1e-10)
    assert fit.fun <= objective(starts[0]) * (1 + 1e-8)
    return d, dict(objective='fixed_participation_sensitivity_squared * mean_full_workload_error',
                  optimized_objective=exact, initial_objective=objective(starts[0]),
                  full_workload_shape=list(W.shape), optimizer='Nelder-Mead, two fixed starts',
                  iterations=int(fit.nit), converged=bool(fit.success), d0_fixed=1.)


@lru_cache(None)
def bias_matrices(n=250, bands=4, beta=.9):
    assert (n, bands, beta) == (250, 4, .9)
    W = momentum_bias_workload(n, beta)
    initial = exp2_matrices('momentum_bandinvmf', n, bands, beta)[0]
    d, metadata = optimize_bias_filter(W, initial)
    return d, inverse_strategy(d, n), W, metadata


def build_matrices(noise, total_steps, num_bands, beta1):
    if noise == 'momentum_bias_bandinvmf':
        return bias_matrices(total_steps, num_bands, beta1)[:3]
    d, strategy, w = exp2_matrices(noise, total_steps, num_bands, beta1)
    return d, strategy, materialize(w, total_steps)
