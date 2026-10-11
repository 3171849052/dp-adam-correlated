"""Adam-aware objectives. Cache once per full horizon/lambda, with C_clip=1."""
from functools import lru_cache
import json
import numpy as np
from scipy.optimize import minimize
from exp10 import RESULTS
from exp10.config import save_json, array_hash
from exp10.bandinvmf import bias_matrices, inverse_strategy, workload_error
from exp10.noise import materialize
from exp10.privacy import fixed_epoch_sensitivity, target_mu

def weights(n):
    t=np.arange(1,n+1); lag=t[:,None]-t[None,:]
    return np.where(lag>=0,.001*.999**np.maximum(lag,0)/(1-.999**t)[:,None],0.)
def components(d,n,W=None):
    W=bias_matrices(n)[2] if W is None else W
    D=materialize(d,n); S=inverse_strategy(d,n)
    sensitivity=fixed_epoch_sensitivity(S,5,n//5)
    sigma=sensitivity/(1000*target_mu(8.,1e-5))
    covariance=sigma**2*(D@D.T); w=weights(n)
    j1=sensitivity**2*workload_error(W,d)
    j2=2*float(np.sum((w@covariance**2)*w))/n
    return dict(J1=j1,J2=j2,sensitivity=sensitivity,sigma_avg_C1=sigma,
                covariance_diagonal_mean=float(np.diag(covariance).mean()),
                covariance_lag1_mean=float(np.diag(covariance,1).mean()))
def optimize(n,lam):
    base,_,W,base_meta=bias_matrices(n); baseline=components(base,n,W)
    # Precompute full Exp9 W1 quadratic and compact 4-band DD^T diagonals.
    shifted=np.stack([np.pad(W[:,lag:],((0,0),(0,lag))) for lag in range(4)])
    Q=np.einsum('aij,bij->ab',shifted,shifted)/n
    w=weights(n); H=w.T@w
    mu=target_mu(8.,1e-5)
    idx=np.arange(n//5)[:,None]+(n//5)*np.arange(5)[None,:]
    def objective(x):
        d=np.r_[1.,x]
        with np.errstate(over='ignore',invalid='ignore'):
            S=inverse_strategy(d,n)
            if not np.isfinite(S).all() or np.max(np.abs(S))>1e8: return 1e100
            cols=S[:,idx]; gram=np.einsum('njk,njl->jkl',cols,cols)
            sensitivity2=np.max(np.abs(gram).sum(axis=(1,2)))
            j1=sensitivity2*(d@Q@d)
            # DD^T is banded, including its truncated startup rows.
            square_sum=0.
            for lag in range(4):
                row=np.arange(n-lag)
                diagonal=np.array([np.dot(d[:min(r+1,4-lag)],d[lag:lag+min(r+1,4-lag)]) for r in row])
                square_sum+=(1 if lag==0 else 2)*np.dot(np.diag(H,lag),diagonal**2)
            j2=2/n*(sensitivity2/(1000*mu)**2)**2*square_sum
            return float(j1/baseline['J1']+lam*j2/baseline['J2'])
    fits=[minimize(objective,x,method='Nelder-Mead',options=dict(maxiter=1600,xatol=1e-9,fatol=1e-7))
          for x in (base[1:]/base[0],np.zeros(3))]
    fit=min(fits,key=lambda r:r.fun); assert fit.success,fit.message
    d=np.r_[1.,fit.x]; actual=components(d,n,W)
    assert np.isclose(fit.fun,actual['J1']/baseline['J1']+lam*actual['J2']/baseline['J2'],rtol=1e-10)
    assert fit.fun<=objective(base[1:])*(1+1e-8)
    return d,dict(horizon=n,lambda_value=lam,C_clip_objective=1.,**actual,baseline=baseline,
        normalized_objective=float(fit.fun),optimizer='Exp9 Nelder-Mead, two fixed starts',iterations=int(fit.nit),
        converged=bool(fit.success),base_coefficients=base.tolist(),base_optimization=base_meta)
@lru_cache(None)
def get_matrix(n,lam=None):
    assert n in (225,250,310)
    tag='base' if lam is None else f'lambda_{lam:g}'
    folder=RESULTS/'matrix_cache'; path=folder/f'T{n}_{tag}.npz'; meta_path=path.with_suffix('.json')
    if path.exists():
        with np.load(path) as data: d=data['coefficients'].copy(); S=data['strategy'].copy(); W=data['workload'].copy()
        meta=json.loads(meta_path.read_text())
        assert meta['strategy_sha256']==array_hash(S) and meta['coefficients_sha256']==array_hash(d)
        assert meta['workload_sha256']==array_hash(W)
    else:
        if lam is None:
            d,S,W,original=bias_matrices(n); meta=dict(horizon=n,lambda_value=None,**components(d,n,W),base_optimization=original)
        else:
            d,meta=optimize(n,lam); S=inverse_strategy(d,n); W=bias_matrices(n)[2]
        folder.mkdir(parents=True,exist_ok=True)
        np.savez(path,coefficients=d,strategy=S,workload=W)
        meta.update(strategy_sha256=array_hash(S),coefficients_sha256=array_hash(d),workload_sha256=array_hash(W))
        save_json(meta_path,meta)
    return d,S,W,meta
def prepare_matrices():
    for n in (225,250,310):
        get_matrix(n)
        for lam in (.1,.3,1.,3.): get_matrix(n,lam)
