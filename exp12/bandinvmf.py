"""Exp2 filters plus full-matrix optimization for non-Toeplitz momentum bias."""
from functools import lru_cache
import json
from exp12 import RESULTS
from exp12.config import save_json, array_hash
import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from exp12.noise import BandInvMFNoise, materialize, build_matrices as exp2_matrices
from exp12.privacy import fixed_epoch_sensitivity


def original_workload(n, beta1):
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


def optimize_bias_filter(W, initial, k=5, spacing=62):
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


def momentum_bias_workload(n, beta1, beta2=.999):
    t = np.arange(1, n + 1)
    lag = t[:, None] - t[None, :]
    moments = np.where(lag >= 0, beta1 ** np.maximum(lag, 0), 0.)
    moments *= (np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))[:, None]
    return np.tril(np.cumsum(moments, axis=0))


@lru_cache(None)
def bias_matrices(n=225, bands=4, beta=.9):
    assert n in (225, 250) and (bands, beta) == (4, .9)
    path = RESULTS / 'matrices' / f'T{n}'
    if (path / 'matrices.npz').exists():
        with np.load(path / 'matrices.npz') as values:
            d, S, W = (values[key].copy() for key in ('coefficients', 'strategy', 'workload'))
        metadata = json.loads((path / 'optimization.json').read_text())
        for key, value in [('coefficients', d), ('strategy', S), ('workload', W)]:
            assert array_hash(value) == metadata[key + '_sha256']
        return d, S, W, metadata
    W = momentum_bias_workload(n, beta)
    initial = exp2_matrices('momentum_bandinvmf', n, bands, beta)[0]
    d, metadata = optimize_bias_filter(W, initial, spacing=n//5)
    S = inverse_strategy(d, n)
    old_W = original_workload(n, beta)
    old_d, old_metadata = optimize_bias_filter(old_W, initial, spacing=n//5)
    old_S = inverse_strategy(old_d, n)
    def metrics(coefficients, strategy):
        sensitivity = fixed_epoch_sensitivity(strategy, 5, n//5)
        return dict(coefficients=coefficients.tolist(), sensitivity=sensitivity,
                    old_workload_error=workload_error(old_W, coefficients),
                    new_workload_error=workload_error(W, coefficients),
                    old_objective=sensitivity**2*workload_error(old_W, coefficients),
                    new_objective=sensitivity**2*workload_error(W, coefficients))
    metadata.update(horizon=n, beta1=beta, beta2=.999, coefficients_sha256=array_hash(d),
                    strategy_sha256=array_hash(S), workload_sha256=array_hash(W),
                    old_filter=metrics(old_d, old_S), new_filter=metrics(d, S),
                    old_optimization=old_metadata, optimality='Exp9 numerical local optimization, no global optimum claim')
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / 'matrices.npz', coefficients=d, strategy=S, workload=W,
             old_coefficients=old_d, old_strategy=old_S, old_workload=old_W)
    save_json(path / 'optimization.json', metadata)
    return d, S, W, metadata


def build_matrices(noise, total_steps, num_bands, beta1):
    assert noise == 'momentum_bias_bandinvmf'
    return bias_matrices(total_steps, num_bands, beta1)[:3]
