"""Float64 four-band geometry, GDP calibration and exact coordinate Gram reduction.

No model, optimizer step, autograd, or historical experiment imports.
"""
import hashlib
import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import brentq, minimize
from scipy.special import log_ndtr
import torch

BETA1, BETA2, EPS = .9, .999, 1e-8
MAX_INVERSE_ABS = 1e8  # Exp9's existing numerical search domain
SEEDS = tuple(range(20261111, 20261121))
HORIZONS = (225, 250, 310)
METHODS = ('iid', 'exp9_momentum_bias', 'expected_self_consistent',
           'actual_self_consistent')


def configure_torch():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def materialize(d, n):
    column = np.zeros(n, dtype=np.float64)
    column[:min(len(d), n)] = d[:n]
    return toeplitz(column, np.r_[d[0], np.zeros(n - 1)])


def inverse_strategy(d, n):
    d = np.asarray(d, dtype=np.float64)
    inverse = np.zeros(n, dtype=np.float64)
    inverse[0] = 1 / d[0]
    # Same inverse Toeplitz recurrence as Exp9. Overflow is an error, never a retry.
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for t in range(1, n):
            width = min(t, len(d) - 1)
            inverse[t] = -np.dot(d[1:width + 1], inverse[t-width:t][::-1]) / d[0]
    return toeplitz(inverse, np.r_[inverse[0], np.zeros(n - 1)])


def sensitivity_squared(C, epochs=5):
    n = len(C)
    assert n % epochs == 0
    spacing = n // epochs
    indices = np.arange(spacing)[:, None] + spacing * np.arange(epochs)[None, :]
    columns = C[:, indices]
    grams = np.einsum('njk,njl->jkl', columns, columns)
    return float(np.abs(grams).sum(axis=(1, 2)).max())


def target_mu(epsilon=8., delta=1e-5):
    def residual(mu):
        a, b = -epsilon / mu + mu / 2, -epsilon / mu - mu / 2
        log_a = log_ndtr(a)
        return np.exp(log_a) * (-np.expm1(epsilon + log_ndtr(b) - log_a)) - delta
    return float(brentq(residual, 1e-6, 100.))


def momentum_initial():
    """Exp9's four-band inverse square root of cumulative momentum.

    Local NumPy implementation of the same coefficient recurrence used by
    jax_privacy.toeplitz.banded_inverse_square_root_noising_coefs.
    """
    w = np.cumsum(BETA1 ** np.arange(4, dtype=np.float64))
    root = np.zeros(4, dtype=np.float64)
    root[0] = np.sqrt(w[0])
    for j in range(1, 4):
        root[j] = (w[j] - np.dot(root[1:j], root[1:j][::-1])) / (2 * root[0])
    return inverse_strategy(root, 4)[:, 0]


def moment_matrix(n, beta):
    t = np.arange(1, n + 1)
    lag = t[:, None] - t[None, :]
    return np.where(lag >= 0, (1-beta) * beta ** np.maximum(lag, 0), 0.) / (1-beta**t[:, None])


def fit_multistart(objective, maxiter=1600):
    """Exactly Exp9's two starts and Nelder-Mead tolerances; d0 fixed at one."""
    starts = (momentum_initial()[1:], np.zeros(3, dtype=np.float64))
    fits, records = [], []
    for x in starts:
        fit = minimize(objective, x.copy(), method='Nelder-Mead',
                       options=dict(maxiter=maxiter, xatol=1e-9, fatol=1e-7))
        fits.append(fit)
        records.append(dict(initial=x.tolist(), d=np.r_[1., fit.x].tolist(),
                            objective=float(fit.fun), iterations=int(fit.nit),
                            evaluations=int(fit.nfev), success=bool(fit.success),
                            message=str(fit.message)))
    best = min(fits, key=lambda f: f.fun)
    assert best.success, best.message
    return np.r_[1., best.x], records


def exp9_filter(n, maxiter=1600):
    # Exp9 omits the global (1-beta1); keep that scaling for solver parity.
    W = np.tril(np.ones((n, n))) @ moment_matrix(n, BETA1) / (1-BETA1)
    shifts = np.stack([np.pad(W[:, i:], ((0, 0), (0, i))) for i in range(4)])
    Q = np.einsum('aij,bij->ab', shifts, shifts) / n
    def objective(x):
        d = np.r_[1., x]
        C = inverse_strategy(d, n)
        value = sensitivity_squared(C) * float(d @ Q @ d)
        return value if np.abs(C).max() <= MAX_INVERSE_ABS else 1e100
    return fit_multistart(objective, maxiter)


def innovations(seed, n, k, device):
    """CPU PCG64 draw, copied to GPU: seed means the same Xi on every GPU."""
    x = np.random.default_rng(seed).standard_normal((n, k), dtype=np.float64)
    return torch.as_tensor(x, device=device), digest(x)


class Evaluator:
    def __init__(self, xi, epsilon=8., delta=1e-5, clip=1., batch=1000.):
        assert xi.ndim == 2 and xi.dtype == torch.float64 and not xi.requires_grad
        self.xi = xi.clone()  # owns the immutable innovation matrix for this fit
        self.n, self.k = xi.shape
        assert self.n >= 5 and self.n % 5 == 0
        self.mu = target_mu(epsilon, delta)
        self.scale = clip / (batch * self.mu)
        self.M = torch.as_tensor(moment_matrix(self.n, BETA1), device=xi.device)
        self.B = torch.as_tensor(moment_matrix(self.n, BETA2), device=xi.device)
        t = torch.arange(self.n, device=xi.device)
        self.H = (self.n - torch.maximum(t[:, None], t[None, :])).to(torch.float64)
        self.calls = 0

    def geometry(self, d):
        C = inverse_strategy(d, self.n)
        sensitivity2 = sensitivity_squared(C)
        sigma = self.scale * np.sqrt(sensitivity2)
        assert np.isfinite(sigma) and sigma > 0
        return C, sensitivity2, sigma

    def gram_q(self, a):
        """G = sum_c W_c^T W_c = M^T [(L^T L) * (A A^T)] M.

        Q[i,j] = sum_{r < T-max(i,j)} G[r+i,r+j] / (T*K).
        A may have one column for the expected-v comparator; its average is
        the same as replicating this column K times.
        """
        G = self.M.T @ (self.H * (a @ a.T)) @ self.M
        Q = torch.empty((4, 4), dtype=torch.float64, device=a.device)
        for i in range(4):
            for j in range(4):
                size = self.n - max(i, j)
                Q[i, j] = G[i:i+size, j:j+size].diagonal().sum()
        return Q / (self.n * a.shape[1])

    def evaluate(self, d, mode='actual', details=False):
        """Every call rebuilds N(d), vhat(d), A(d), Gram(d), and Q(d).

        J is a Frobenius workload functional, not the realized squared error.
        Only the explicitly named expected comparator uses E[N^2].
        """
        assert mode in ('actual', 'expected')
        d = np.asarray(d, dtype=np.float64)
        assert d.shape == (4,) and d[0] == 1.
        C, sens2, sigma = self.geometry(d)
        noise = self.xi * float(d[0])
        for lag in range(1, 4):
            noise[lag:] += float(d[lag]) * self.xi[:-lag]
        noise *= sigma
        if mode == 'actual':
            vhat = self.B @ noise.square()
        else:
            variance = sigma**2 * np.cumsum(d**2)[np.minimum(np.arange(self.n), 3)]
            vhat = self.B @ torch.as_tensor(variance[:, None], device=self.xi.device)
        a = 1 / (torch.sqrt(vhat) + EPS)
        Q = self.gram_q(a)
        dt = torch.as_tensor(d, device=self.xi.device)
        J = float((sigma**2 * (dt @ Q @ dt)).item())
        assert np.isfinite(J) and J > 0
        self.calls += 1
        if not details:
            return J
        return dict(J=J, Q=Q, noise=noise, vhat=vhat, a=a,
                    sigma=sigma, sensitivity_squared=sens2, C=C)

    def fit(self, mode, maxiter=1600, trace=None):
        before = self.calls
        def objective(x):
            d = np.r_[1., x]
            # Keep Exp9's inverse-amplification domain. Even rejected candidates
            # rebuild N, vhat and Q first; no frozen-Q evaluation or resampling.
            result = self.evaluate(d, mode, details=True)
            admissible = bool(np.abs(result['C']).max() <= MAX_INVERSE_ABS)
            value = result['J'] if admissible else 1e100
            if trace is not None:
                trace(dict(evaluation=self.calls-before, d=d.tolist(), J=result['J'],
                           optimizer_value=value, admissible=admissible, mode=mode))
            return value
        d, starts = fit_multistart(objective, maxiter)
        assert np.abs(inverse_strategy(d, self.n)).max() <= MAX_INVERSE_ABS
        return d, dict(mode=mode, starts=starts, evaluations=self.calls-before,
                       objective_at_solution=self.evaluate(d, mode))

    def metrics(self, d):
        result = self.evaluate(d, 'actual', details=True)
        # Full pure-noise Adam, using N for m and N^2 for v, both bias corrected.
        mhat = self.M @ result['noise']
        cumulative = torch.cumsum(mhat * result['a'], dim=0)
        J_real = float(cumulative.square().mean().item())
        v = result['vhat']
        quantiles = torch.tensor([0., .01, .05, .5, .95, .99, 1.],
                                 dtype=torch.float64, device=v.device)
        def summary(x):
            return dict(mean=float(x.mean().item()), std=float(x.std(correction=0).item()),
                        quantiles=dict(zip(('min', 'p01', 'p05', 'p50', 'p95', 'p99', 'max'),
                                           torch.quantile(x.flatten(), quantiles).cpu().tolist())))
        C = result['C']
        marginal = result['sigma'] * np.sqrt(np.cumsum(np.asarray(d)**2)[np.minimum(np.arange(self.n), 3)])
        metrics = dict(d=np.asarray(d).tolist(), J=result['J'], J_real=J_real,
                       innovation_std=result['sigma'], sensitivity=np.sqrt(result['sensitivity_squared']),
                       inverse_max_abs=float(np.abs(C).max()),
                       inverse_spectral_norm=float(np.linalg.norm(C, ord=2)),
                       inverse_frobenius_norm=float(np.linalg.norm(C)),
                       marginal_noise_std_first=float(marginal[0]),
                       marginal_noise_std_steady=float(marginal[-1]),
                       noise_rms=float(result['noise'].square().mean().sqrt().item()),
                       vhat_all=summary(v), vhat_final=summary(v[-1]),
                       per_step_vhat_quantiles=torch.quantile(v, quantiles, dim=1).T.cpu().tolist())
        return metrics, result
