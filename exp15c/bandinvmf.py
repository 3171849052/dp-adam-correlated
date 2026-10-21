"""Exp15b FIR parameterization and full-matrix optimization for Momentum."""
from functools import lru_cache
import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from exp15c.noise import BandInvMFNoise, materialize, build_matrices as exp2_matrices
from exp15c.privacy import fixed_epoch_sensitivity


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


def optimize_filter(W, initial, k=5, spacing=62):
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



def momentum_workload(n,beta=.9):
    t=np.arange(n);lag=t[:,None]-t[None,:]
    return np.where(lag>=0,(1-beta**(np.maximum(lag,0)+1))/(1-beta),0.)


def native_matrices(n):
    from exp15c import RESULTS
    path=RESULTS/'mf'/f'momentum_{n}.npz'
    with np.load(path) as f:
        return tuple(f[k].copy() for k in ('coefficients','strategy','workload'))


def prepare():
    """Freeze the three horizons before any model training; calibrate anew."""
    from exp15c import RESULTS
    from exp15c.config import Trial,save_json,array_hash,file_hash
    from exp15c.privacy import calibrate,epsilon_from_mu
    from exp15c.support import historical_matrices,write_csv
    directory=RESULTS/'mf';directory.mkdir(exist_ok=True)
    comparisons=[]
    for n,task,phase in ((225,'cv','search'),(250,'cv','final'),(310,'nlp','search')):
        cfg=Trial(task,'momentum-mf-native',.01,1.,smoke_phase=phase)
        path=directory/f'momentum_{n}.npz';meta_path=path.with_suffix('.json')
        if path.exists():
            d,S,W=native_matrices(n)
            import json
            meta=json.loads(meta_path.read_text())
            if meta['matrix_sha256']!=file_hash(path): raise ValueError('MF file changed')
        else:
            W=momentum_workload(n)
            initial=exp2_matrices('momentum_bandinvmf',n,4,.9)[0]
            d,meta=optimize_filter(W,initial,spacing=n//5)
            S=inverse_strategy(d,n)
            np.savez(path,coefficients=d,strategy=S,workload=W,actual_workload=.1*W)
            meta.update(matrix_sha256=file_hash(path),coefficients=d.tolist(),strategy_sha256=array_hash(S))
            save_json(meta_path,meta)
        calibration=calibrate(S,cfg.protocol())
        calibration['verified_full_epsilon']=epsilon_from_mu(calibration['sensitivity']/calibration['innovation_std_sum'],1e-5)
        save_json(directory/f'privacy_{n}.json',calibration)
        old_d,old_S,old_W,source=historical_matrices(cfg)
        native=fixed_epoch_sensitivity(S,5,n//5)**2*workload_error(W,d)
        old=fixed_epoch_sensitivity(old_S,5,n//5)**2*workload_error(W,old_d)
        mu=calibration['target_mu']
        comparisons.append(dict(task=task,total_steps=n,clip=1.,epsilon=8.,delta=1e-5,
            native_objective=native,historical_objective_on_momentum=old,relative_reduction=1-native/old,
            native_design_error=workload_error(W,d),historical_bias_design_error=workload_error(old_W,old_d),
            native_sensitivity=calibration['sensitivity'],historical_sensitivity=fixed_epoch_sensitivity(old_S,5,n//5),
            native_innovation_std_sum_C1=calibration['innovation_std_sum'],
            historical_innovation_std_sum_C1=fixed_epoch_sensitivity(old_S,5,n//5)/mu,
            native_actual_workload_MSE_C1=native*.1**2/mu**2/1000**2,
            historical_actual_workload_MSE_C1=old*.1**2/mu**2/1000**2,
            mse_units='cumulative momentum noise per coordinate; excludes lr^2; scales with C^2',historical_source=source))
    write_csv(RESULTS/'mf_comparison.csv',comparisons)
    return comparisons


if __name__=='__main__':
    import json
    print(json.dumps(prepare(),indent=2))
