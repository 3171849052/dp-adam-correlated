"""Adam-aware objectives. Cache once per full horizon/lambda, with C_clip=1."""
from functools import lru_cache
import json
import numpy as np
from scipy.optimize import minimize
from exp10b import ROOT, BASE, RESULTS
from exp10b.config import save_json, array_hash, file_hash, LAMBDAS
from exp10b.bandinvmf import inverse_strategy, workload_error
from exp10b.noise import materialize
from exp10b.privacy import fixed_epoch_sensitivity, target_mu

@lru_cache(None)
def baseline_matrix(n):
    """Load Exp9's actual D0, never run its optimizer for lambda zero."""
    ref=json.loads((BASE/'exp9_matrix_references.json').read_text())[str(n)]
    assert file_hash(ROOT/ref['path'])==ref['sha256']
    assert file_hash(ROOT/ref['summary'])==ref['summary_sha256']
    with np.load(ROOT/ref['path']) as data:
        d,S,W=(data[k].copy() for k in ('coefficients','strategy','workload'))
    original=json.loads((ROOT/ref['summary']).read_text())['matrix_optimization']
    return d,S,W,original

def weights(n):
    t=np.arange(1,n+1); lag=t[:,None]-t[None,:]
    return np.where(lag>=0,.001*.999**np.maximum(lag,0)/(1-.999**t)[:,None],0.)
def components(d,n,W=None):
    W=baseline_matrix(n)[2] if W is None else W
    D=materialize(d,n); S=inverse_strategy(d,n)
    sensitivity=fixed_epoch_sensitivity(S,5,n//5)
    sigma=sensitivity/(1000*target_mu(8.,1e-5))
    covariance=sigma**2*(D@D.T); w=weights(n)
    j1=sensitivity**2*workload_error(W,d)
    j2=2*float(np.sum((w@covariance**2)*w))/n
    j2_diag=2*float(np.sum(w**2@np.diag(covariance)**2))/n
    j2_cross=4*float(np.sum((w@np.triu(covariance**2,1))*w))/n
    assert np.isclose(j2,j2_diag+j2_cross,rtol=1e-12)
    return dict(J1=j1,J2=j2,sensitivity=sensitivity,sigma_avg_C1=sigma,
                J2_diag=j2_diag,J2_cross=j2_cross,R_corr=j2/j2_diag,
                covariance_diagonal_mean=float(np.diag(covariance).mean()),
                covariance_lag1_mean=float(np.diag(covariance,1).mean()))
def optimize(n,lam):
    base,_,W,base_meta=baseline_matrix(n); baseline=components(base,n,W)
    if lam==0:
        return base.copy(),dict(horizon=n,lambda_value=0.,C_clip_objective=1.,**baseline,
            baseline=baseline,normalized_objective=1.,optimizer='Exp9 stored D0, no optimization',
            converged=True,iterations=0,base_coefficients=base.tolist(),base_optimization=base_meta)
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
                diagonal=np.cumsum(d[:4-lag]*d[lag:])[np.minimum(row,3-lag)]
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
def get_matrix(n,lam=0.):
    assert n in (225,250,310) and lam in LAMBDAS
    tag=f'lambda_{lam:g}'
    folder=RESULTS/'matrix_cache'; path=folder/f'T{n}_{tag}.npz'; meta_path=path.with_suffix('.json')
    if path.exists():
        with np.load(path) as data: d=data['coefficients'].copy(); S=data['strategy'].copy(); W=data['workload'].copy()
        meta=json.loads(meta_path.read_text())
        assert meta['strategy_sha256']==array_hash(S) and meta['coefficients_sha256']==array_hash(d)
        assert meta['workload_sha256']==array_hash(W)
        for name in ('inverse','covariance_C1','marginal_variance_C1'):
            with np.load(path) as data: assert array_hash(data[name])==meta[name+'_sha256']
    else:
        d,meta=optimize(n,lam)
        S=baseline_matrix(n)[1].copy() if lam==0 else inverse_strategy(d,n)
        W=baseline_matrix(n)[2].copy()
        D=materialize(d,n); covariance=meta['sigma_avg_C1']**2*(D@D.T)
        meta.update(J1_ratio=meta['J1']/meta['baseline']['J1'],J2_ratio=meta['J2']/meta['baseline']['J2'],
                    exp9_reference=json.loads((BASE/'exp9_matrix_references.json').read_text())[str(n)])
        folder.mkdir(parents=True,exist_ok=True)
        arrays=dict(coefficients=d,strategy=S,workload=W,inverse=D,covariance_C1=covariance,
                    marginal_variance_C1=np.diag(covariance))
        np.savez(path,**arrays)
        for name,value in arrays.items(): meta[name+'_sha256']=array_hash(value)
        meta.update(strategy_sha256=array_hash(S),coefficients_sha256=array_hash(d),workload_sha256=array_hash(W))
        save_json(meta_path,meta)
    return d,S,W,meta
def prepare_matrices():
    for n in (225,250,310):
        for lam in LAMBDAS: get_matrix(n,lam)
