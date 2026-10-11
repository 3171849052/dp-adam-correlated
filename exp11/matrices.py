"""Four-tap BandInvMF: full momentum workload + Gaussian-square second-moment MSE.

The compact covariance diagonals include truncated startup rows. B maps squared
taps to the mean of the bias-corrected second moment. Thus B.T B supplies its
bias-square quadratic; the variance uses the four diagonals of w.T w.
"""
import json
import numpy as np
from scipy.optimize import minimize
from exp11 import RESULTS
from exp11.config import T, CLIP, array_hash, lambda_tag, save_json
from exp11.bandinvmf import momentum_bias_workload, inverse_strategy
from exp11.noise import build_matrices, materialize
from exp11.privacy import fixed_epoch_sensitivity, target_mu


def weights(n, beta2=.999):
    t = np.arange(1, n + 1)
    lag = t[:, None] - t[None, :]
    return np.where(lag >= 0, (1 - beta2) * beta2 ** np.maximum(lag, 0)
                    / (1 - beta2 ** t)[:, None], 0.)


class Objective:
    def __init__(self, n=T, clip=CLIP):
        self.n, self.clip = n, clip
        self.W = momentum_bias_workload(n, .9)
        shifted = np.stack([np.pad(self.W[:, lag:], ((0, 0), (0, lag))) for lag in range(4)])
        self.Q = np.einsum('aij,bij->ab', shifted, shifted) / n
        w = weights(n)
        H = w.T @ w
        self.h_diagonals = [np.diag(H, lag) for lag in range(4)]
        B = np.stack([w[:, lag:].sum(axis=1) for lag in range(4)], axis=1)
        self.bias_Q = B.T @ B / n
        self.indices = np.arange(n // 5)[:, None] + (n // 5) * np.arange(5)[None, :]
        self.mu = target_mu(8., 1e-5)
        self.iid = self.components(np.array([1., 0., 0., 0.]))

    def components(self, d):
        S = inverse_strategy(d, self.n)
        # Identical Exp9 numerical stability domain and absolute Gram bound.
        if not np.isfinite(S).all() or np.max(np.abs(S)) > 1e8:
            return None
        columns = S[:, self.indices]
        grams = np.einsum('njk,njl->jkl', columns, columns)
        sensitivity2 = np.max(np.abs(grams).sum(axis=(1, 2)))
        sigma2 = self.clip ** 2 * sensitivity2 / (1000 * self.mu) ** 2
        square_sum = 0.
        for lag in range(4):
            startup = np.cumsum(d[:4-lag] * d[lag:])
            diagonal = np.r_[startup[:-1], np.full(self.n - lag - len(startup) + 1, startup[-1])]
            square_sum += (1 if lag == 0 else 2) * np.dot(self.h_diagonals[lag], diagonal ** 2)
        variance = 2 * sigma2 ** 2 * square_sum / self.n
        bias_sq = sigma2 ** 2 * (d ** 2 @ self.bias_Q @ d ** 2)
        return dict(J1=float(sensitivity2 * (d @ self.Q @ d)),
                    J2_variance=float(variance), J2_bias_sq=float(bias_sq),
                    J2_mse=float(variance + bias_sq), sensitivity=float(np.sqrt(sensitivity2)),
                    sigma_avg=float(np.sqrt(sigma2)), C_clip=self.clip)

    def value(self, x, lam):
        with np.errstate(over='ignore', invalid='ignore'):
            actual = self.components(np.r_[1., x])
        if actual is None:
            return 1e100
        return actual['J1'] / self.iid['J1'] + lam * actual['J2_mse'] / self.iid['J2_mse']


def dense_components(d, n=T, clip=CLIP):
    W = momentum_bias_workload(n, .9)
    D = materialize(d, n)
    sensitivity = fixed_epoch_sensitivity(inverse_strategy(d, n), 5, n // 5)
    sigma = clip * sensitivity / (1000 * target_mu(8., 1e-5))
    covariance = sigma ** 2 * (D @ D.T)
    w = weights(n)
    variance = 2 * float(np.sum((w @ covariance ** 2) * w)) / n
    bias_sq = float(np.mean((w @ np.diag(covariance)) ** 2))
    return dict(J1=sensitivity ** 2 * float(np.sum((W @ D) ** 2) / n),
                J2_variance=variance, J2_bias_sq=bias_sq, J2_mse=variance + bias_sq,
                sensitivity=sensitivity, sigma_avg=sigma, C_clip=clip)


def get_matrix(lam, neighbors=()):
    folder = RESULTS / 'matrices' / lambda_tag(lam)
    path = folder / 'matrices.npz'
    if path.exists():
        with np.load(path) as cached:
            d, S, W = (cached[key].copy() for key in ('coefficients', 'strategy', 'workload'))
        metadata = json.loads((folder / 'optimization.json').read_text())
        for key, value in [('coefficients', d), ('strategy', S), ('workload', W)]:
            assert array_hash(value) == metadata[key + '_sha256']
        return d, S, W, metadata
    kernel = Objective()
    bisr = build_matrices('momentum_bandinvmf', T, 4, .9)[0]
    starts = [('Exp9 BISR', bisr[1:] / bisr[0]), ('IID', np.zeros(3))]
    starts += [(f'neighbor lambda={value:.17g}', np.asarray(d)[1:]) for value, d in neighbors]
    fits = [minimize(kernel.value, x, args=(lam,), method='Nelder-Mead',
                     options=dict(maxiter=1600, xatol=1e-9, fatol=1e-7)) for _, x in starts]
    fit = min(fits, key=lambda r: r.fun)
    assert fit.success, fit.message
    d = np.r_[1., fit.x]
    actual = dense_components(d)
    compact = kernel.components(d)
    for key in actual:
        assert np.isclose(actual[key], compact[key], rtol=1e-10, atol=1e-20), key
    normalized = {f'normalized_{key}': actual[key] / kernel.iid['J1' if key == 'J1' else 'J2_mse']
                  for key in ('J1', 'J2_variance', 'J2_bias_sq', 'J2_mse')}
    assert np.isclose(fit.fun, normalized['normalized_J1'] + lam * normalized['normalized_J2_mse'], rtol=1e-10)
    metadata = dict(horizon=T, lambda_value=lam, **actual, **normalized, iid=kernel.iid,
                    normalized_objective=float(fit.fun), coefficients=d.tolist(), d0_fixed=1.,
                    optimizer='Exp9 Nelder-Mead; BISR, IID, available neighboring lambda starts',
                    optimality='Numerical local optimum; no global optimality claim',
                    starts=[dict(name=name, coefficients=np.r_[1., x].tolist(),
                                 initial_objective=float(kernel.value(x, lam)),
                                 final_objective=float(r.fun), iterations=int(r.nit),
                                 converged=bool(r.success), message=str(r.message))
                            for (name, x), r in zip(starts, fits)])
    S, W = inverse_strategy(d, T), kernel.W
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(path, coefficients=d, strategy=S, inverse=materialize(d, T), workload=W,
             second_moment_weights=weights(T))
    metadata.update(coefficients_sha256=array_hash(d), strategy_sha256=array_hash(S), workload_sha256=array_hash(W))
    save_json(folder / 'optimization.json', metadata)
    return d, S, W, metadata
