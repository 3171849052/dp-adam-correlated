"""Data-independent L-BFGS designs, full workload objectives and matrix audits."""
import time
import numpy as np
import torch
from scipy.linalg import toeplitz,solve_triangular
from exp16d import RESULTS
from exp16d.config import save_json,file_hash,array_hash
from exp16d.privacy import fixed_epoch_sensitivity,target_mu,epsilon_from_mu

def workload(name,n):
    t=np.arange(1,n+1);lag=t[:,None]-t[None,:]
    if name=='identity':return np.eye(n)
    if name=='sgd':return np.tril(np.ones((n,n)))
    if name=='momentum':return np.where(lag>=0,1-.9**(np.maximum(lag,0)+1),0.)
    if name=='momentum-bias':
        moments=np.where(lag>=0,.9**np.maximum(lag,0),0.)/(1-.9**t)[:,None]
        return np.tril(np.cumsum(moments,axis=0))
    raise ValueError(name)
def materialize(d,n):return toeplitz(np.pad(d,(0,n-len(d))),np.r_[d[0],np.zeros(n-1)])
def bisr(a,bands=4):
    # The paper's truncated inverse square root, normalized only in strategy scale.
    s=np.zeros(bands);s[0]=np.sqrt(a[0])
    for j in range(1,bands):s[j]=(a[j]-s[1:j]@s[1:j][::-1])/(2*s[0])
    d=solve_triangular(materialize(s,bands),np.eye(bands),lower=True)[:,0]
    return d/d[0]
def tensor_toeplitz(c,n):
    idx=torch.arange(n)[:,None]-torch.arange(n)[None,:]
    return torch.where(idx>=0,c[idx.clamp_min(0)],0.)

def audit_arrays(A,C,D):
    n=len(A);identity=np.eye(n);k=n//5
    inverse_error=max(np.max(np.abs(C@D-identity)),np.max(np.abs(D@C-identity)))
    if inverse_error>1e-7:raise ValueError('Inverse mismatch')
    B=A@D
    sens=fixed_epoch_sensitivity(C,5,k)
    # Independent columns/direction upper bound, no Gram reuse.
    column_bounds=[float(np.sum(np.abs(C[:,j+k*np.arange(5)]).sum(1)**2)) for j in range(k)]
    absolute_bound=np.sqrt(max(column_bounds))
    if sens>absolute_bound*(1+1e-10):raise ValueError('Sensitivity upper bound failed')
    # Toeplitz and nonnegative cases attain the bound with aligned participation vectors.
    nonnegative=bool(np.min(C)>=-1e-12)
    aligned=max(np.linalg.norm(C[:,j+k*np.arange(5)].sum(1)) for j in range(k))
    if nonnegative and not np.isclose(sens,aligned,rtol=1e-9):raise ValueError('Aligned sensitivity mismatch')
    mu=target_mu(8.,1e-5);sigma=sens/mu;actual=epsilon_from_mu(sens/sigma,1e-5)
    covariance=sigma*sigma*(D@D.T)
    whiten=C@covariance@C.T
    if not np.allclose(whiten,sigma*sigma*identity,atol=1e-6,rtol=1e-6):raise ValueError('Covariance mismatch')
    return dict(inverse_max_abs=inverse_error,factorization_max_abs=float(np.max(np.abs(B@C-A))),
        sensitivity=sens,absolute_column_upper_bound=float(absolute_bound),aligned_lower_bound=float(aligned),nonnegative_strategy=nonnegative,
        mean_workload_error=float(np.sum(B*B)/n),objective=float(sens*sens*np.sum(B*B)/n),
        sigma_sum_clip1=sigma,mu=mu,epsilon=actual,delta=1e-5,
        covariance_sha256=array_hash(covariance),mean_gradient_noise_MSE_clip1=float(np.trace(covariance)/n/1e6),
        workload_noise_MSE_clip1=float(sigma*sigma*np.sum(B*B)/n/1e6))
def load(work,mechanism,n):
    if n not in (225,250) or (work,mechanism) not in [('identity','iid'),('sgd','b2lt'),('momentum','b2lt'),('momentum-bias','b2lt')]:raise ValueError('Unapproved matrix')
    from exp16d import ROOT
    import json
    source=ROOT/'exp16a/results'
    p=source/'matrices'/f'{work}_{mechanism}_T{n}.npz'
    acceptance=json.loads((source/'phase1_acceptance.json').read_text())
    if acceptance['status']!='passed':raise ValueError('Historical matrix acceptance failed')
    for f in (p,p.with_suffix('.json')):
        if file_hash(f)!=acceptance['matrix_sha256'][str(f.relative_to(ROOT))]:raise ValueError('Historical matrix hash changed')
    meta=json.loads(p.with_suffix('.json').read_text())
    with np.load(p) as f:arrays={k:f[k].copy() for k in f.files}
    return dict(arrays,path=p,metadata=meta)

def prepare():
    rows=[]
    for n in (225,250):
        for work,mech in [('identity','iid'),('sgd','b2lt'),('momentum','b2lt'),('momentum-bias','b2lt')]:
            a=load(work,mech,n)
            rows.append(dict(workload=work,mechanism=mech,total_steps=n,source=str(a['path']),sha256=file_hash(a['path']),**audit_arrays(a['A'],a['C'],a['D'])))
    save_json(RESULTS/'matrix_audit.json',rows)
    return rows
