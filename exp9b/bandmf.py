"""Four-band strategy optimization for Exp9's exact full workloads."""
from functools import lru_cache
import json
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from exp9b import RESULTS
from exp9b.noise import materialize, inverse_coefficients
from exp9b.privacy import fixed_epoch_sensitivity
from exp9b.config import array_hash, file_hash, save_json


def momentum_bias_workload(n, beta1):
    t = np.arange(1, n+1)
    lag = t[:, None] - t[None, :]
    moments = np.where(lag >= 0, beta1 ** np.maximum(lag, 0), 0.)
    moments /= (1-beta1**t)[:, None]
    return np.tril(np.cumsum(moments, axis=0))


def workload_matrix(noise, n, beta):
    if noise == 'iid': return np.eye(n)
    if noise == 'prefix_bandmf': return np.tril(np.ones((n,n)))
    if noise == 'momentum_bandmf':
        return materialize(np.cumsum(beta**np.arange(n)), n)
    assert noise == 'momentum_bias_bandmf'
    return momentum_bias_workload(n, beta)


def workload_error(W, c):
    # Right multiplication by S^{-1}: filter each complete row backwards.
    WD = lfilter([1.], c, W[:, ::-1], axis=1)[:, ::-1]
    return float(np.sum(WD**2)/len(W))


def optimize_strategy(W):
    n = len(W); k = 5; spacing = n//k
    assert n == k*spacing and spacing > 3
    # Participation columns have disjoint supports, and j=0 has all four
    # entries for each epoch. Thus Exp9's absolute-Gram sensitivity squared
    # is exactly k*(1+||c||^2), including signed strategy coefficients.
    def objective(x):
        c = np.r_[1., x]
        with np.errstate(over='ignore', invalid='ignore'):
            value = k*float(c@c)*workload_error(W,c)
        return value if np.isfinite(value) else 1e100
    starts = (np.zeros(3), np.array([-.9,0.,0.]), np.array([-1.8,.81,0.]))
    fits = [minimize(objective, x, method='Nelder-Mead',
                     options=dict(maxiter=2400, xatol=1e-10, fatol=1e-8)) for x in starts]
    fit = min(fits, key=lambda r:r.fun)
    assert fit.success, fit.message
    c = np.r_[1.,fit.x]; S = materialize(c,n)
    exact = fixed_epoch_sensitivity(S,k,spacing)**2*workload_error(W,c)
    assert np.isclose(exact,fit.fun,rtol=1e-11)
    assert exact <= objective(starts[0])*(1+1e-8)
    return c, dict(objective='Sens(S)^2 * ||W S^{-1}||_F^2 / T',
                  optimized_objective=exact, initial_objective=objective(starts[0]),
                  full_workload_shape=list(W.shape), optimizer='Nelder-Mead, three fixed starts',
                  iterations=int(fit.nit), converged=bool(fit.success), s0_fixed=1.,
                  starts=[s.tolist() for s in starts],start_objectives=[float(f.fun) for f in fits],
                  bandwidth_constraint='strategy S has four bands; D=S^{-1} is dense')


@lru_cache(None)
def matrix_bundle(noise,n=310,bands=4,beta=.9):
    assert n in (225,250,310) and (bands,beta)==(4,.9)
    directory=RESULTS/'matrix_cache'; path=directory/f'{noise}_T{n}.npz'; meta_path=path.with_suffix('.json')
    if path.exists():
        meta=json.loads(meta_path.read_text()); assert file_hash(path)==meta['file_sha256']
        with np.load(path) as v: c,S,W,D=(v[key] for key in ('coefficients','strategy','workload','inverse'))
        assert all(array_hash(v)==meta['array_sha256'][key] for key,v in zip(('coefficients','strategy','workload','inverse'),(c,S,W,D)))
        return c,S,W,meta
    W=workload_matrix(noise,n,beta)
    if noise=='iid': c=np.array([1.]); optimization=dict(optimized_objective=5.,converged=True)
    else: c,optimization=optimize_strategy(W)
    S=materialize(c,n); D=materialize(inverse_coefficients(c,n),n)
    np.testing.assert_allclose(S@D,np.eye(n),atol=2e-11)
    impulse=D[:,0]; error=float(np.sum((W@D)**2)/n)
    meta=dict(noise=noise,T=n,optimization=optimization,sensitivity=fixed_epoch_sensitivity(S,5,n//5),
              workload_error=error,inverse_max_abs_residual=float(np.max(np.abs(S@D-np.eye(n)))),
              unit_innovation_variance_final=float(np.sum(impulse**2)),
              unit_innovation_variance_tail_mean=float(np.mean(np.cumsum(impulse**2)[-n//5:])),
              impulse_last_three=impulse[-3:].tolist(),
              array_sha256={key:array_hash(v) for key,v in zip(('coefficients','strategy','workload','inverse'),(c,S,W,D))})
    directory.mkdir(parents=True,exist_ok=True)
    np.savez(path,coefficients=c,strategy=S,workload=W,inverse=D)
    meta['file_sha256']=file_hash(path);save_json(meta_path,meta)
    return c,S,W,meta


def build_matrices(noise,total_steps,num_bands,beta1):
    return matrix_bundle(noise,total_steps,num_bands,beta1)[:3]


def bias_matrices(n=310,bands=4,beta=.9):
    c,S,W,meta=matrix_bundle('momentum_bias_bandmf',n,bands,beta)
    return c,S,W,meta['optimization']


def prepare_matrices():
    rows=[matrix_bundle(noise,n)[3] for n in (225,250,310)
          for noise in ('iid','prefix_bandmf','momentum_bandmf','momentum_bias_bandmf')]
    save_json(RESULTS/'matrix_validation.json',rows)
    return rows
