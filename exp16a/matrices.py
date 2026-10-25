"""Data-independent L-BFGS designs, full workload objectives and matrix audits."""
import time
import numpy as np
import torch
from scipy.linalg import toeplitz,solve_triangular
from exp16a import RESULTS
from exp16a.config import save_json,file_hash,array_hash
from exp16a.privacy import fixed_epoch_sensitivity,target_mu,epsilon_from_mu

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

def optimize(A,mechanism,k,work):
    n=len(A);torch.set_num_threads(2);W=torch.tensor(A,dtype=torch.float64);eye=torch.eye(n,dtype=torch.float64)
    Q=5;P=k;trace=[];candidates=[]
    if mechanism=='bandinvmf':
        x0=bisr(workload('momentum',n)[:,0] if work=='momentum-bias' else A[:,0])[1:]
        # Non-Toeplitz Momentum-Bias starts from cumulative Momentum's BISR.
    elif mechanism=='b2lt':
        theta=np.arange(1,Q+1)/(Q+1);x0=np.r_[theta,-(1-theta)/(2*Q)]
    else:x0=np.arange(n-1,0,-1)/n
    def build(x):
        phi=torch.tensor(0.,dtype=torch.float64)
        if mechanism=='dtc':
            c=torch.cat((x.new_ones(1),x));gaps=torch.cat((c[:-1]-c[1:],c[-1:]))
            if torch.any(gaps<=0):return None
            phi=-gaps.log().sum();C=tensor_toeplitz(c,n);D=torch.linalg.solve_triangular(C,eye,upper=False)
        else:
            if mechanism=='b2lt':
                th,om=x[:Q],x[Q:];sep=torch.pdist(th[:,None])
                margins=torch.cat((-om,th*(1-th),(1+om.sum()).view(1),(1+(om/(1-th)).sum()).view(1),sep.min().view(1)))
                if torch.any(margins<=0):return None
                phi=-margins.log().sum()
                d=torch.cat((x.new_ones(1),(om[:,None]*th[:,None]**torch.arange(P-1)).sum(0),x.new_zeros(n-P)))
            else:d=torch.cat((x.new_ones(1),x,x.new_zeros(n-4)))
            D=tensor_toeplitz(d,n);C=torch.linalg.solve_triangular(D,eye,upper=False)
        if not torch.isfinite(C).all() or float(C.detach().abs().max())>1e10:return None
        indices=torch.arange(k)[:,None]+k*torch.arange(5)[None,:]
        cols=C[:,indices];grams=torch.einsum('njk,njl->jkl',cols,cols)
        sens2=grams.abs().sum((1,2)).max();err=(W@D).square().sum()/n
        return sens2*err,phi,C,D
    x=torch.tensor(x0,dtype=torch.float64);initial=build(x)
    e0=float(initial[0]);lam=0. if mechanism=='bandinvmf' else 1e-4*e0/len(x0)
    # Select among accepted iterates, plus the prescribed initial and DTC IID.
    def retain(v,iteration):
        with torch.no_grad():
            z=build(torch.tensor(v,dtype=torch.float64))
            if z is not None:
                e,phi,_,_=z;record=dict(iteration=iteration,error=float(e),barrier=float(phi),objective=float(e+lam*phi))
                trace.append(record);candidates.append((float(e),np.asarray(v).copy()))
    retain(x0,0)
    def fun(v):
        z=torch.tensor(v,dtype=torch.float64,requires_grad=True);out=build(z)
        if out is None:return float('inf'),np.zeros_like(v)
        e,phi,_,_=out;f=e+lam*phi
        if not torch.isfinite(f):return float('inf'),np.zeros_like(v)
        f.backward();return float(f.detach()),z.grad.detach().numpy().copy()
    # Feasible Armijo L-BFGS: reject barrier violations before accepting any step.
    # Explicit backtracking avoids scipy treating an infinite proposal as convergence.
    current=x0.copy();f,g=fun(current);history=[];message='iteration limit';converged=False
    for iteration in range(1,1001):
        if np.max(np.abs(g))/e0<1e-7:message='gradient tolerance';converged=True;break
        q=g.copy();pairs=[]
        for dx,dg,rho in reversed(history):
            alpha=rho*np.dot(dx,q);pairs.append(alpha);q-=alpha*dg
        scale=np.dot(history[-1][0],history[-1][1])/np.dot(history[-1][1],history[-1][1]) if history else 1/e0
        direction=q*scale
        for (dx,dg,rho),alpha in zip(history,reversed(pairs)):
            direction+=dx*(alpha-rho*np.dot(dg,direction))
        direction=-direction;slope=float(g@direction)
        if slope>=0:direction=-g/e0;slope=float(g@direction);history=[]
        step=1.;accepted=False
        for _ in range(60):
            proposal=current+step*direction;newf,newg=fun(proposal)
            if np.isfinite(newf) and newf<=f+1e-4*step*slope:
                accepted=True;break
            step*=.5
        if not accepted:message='feasible line search stalled';break
        dx=proposal-current;dg=newg-g;curvature=float(dx@dg)
        if curvature>1e-14*np.linalg.norm(dx)*np.linalg.norm(dg):
            history.append((dx,dg,1/curvature));history=history[-20:]
        improvement=f-newf;current=proposal;f,g=newf,newg;retain(current,iteration)
        if improvement<=1e-12*max(1.,abs(f)) and np.max(np.abs(g))/e0<1e-5:
            message='relative objective and gradient tolerance';converged=True;break
    best_error,best=min(candidates,key=lambda z:z[0]);selected='iterate'
    _,_,C,D=build(torch.tensor(best,dtype=torch.float64));C=C.detach().numpy();D=D.detach().numpy()
    iid_error=float(5*np.sum(A*A)/n)
    if mechanism=='dtc' and iid_error<best_error:C=D=np.eye(n);best_error=iid_error;selected='iid'
    d=D[:,0].copy()
    meta=dict(optimizer='L-BFGS; memory=20; feasible Armijo backtracking',maxiter=1000,
        iterations=iteration,success=converged,message=message,lambda_barrier=lam,
        initial_parameters=x0.tolist(),selected_parameters=best.tolist(),selected=selected,
        initial_error=e0,selected_error=best_error,iid_error=iid_error,trace=trace,
        bandwidth=4 if mechanism=='bandinvmf' else P if mechanism=='b2lt' else n,
        P=P if mechanism=='b2lt' else None,Q=Q if mechanism=='b2lt' else None,
        theta=best[:Q].tolist() if mechanism=='b2lt' else None,omega=best[Q:].tolist() if mechanism=='b2lt' else None)
    if not np.isfinite(best_error) or best_error>e0*(1+1e-9):raise ValueError('Invalid optimization selection')
    return C,D,d,meta

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
def path_for(work,mechanism,n):return RESULTS/'matrices'/f'{work}_{mechanism}_T{n}.npz'
def load(work,mechanism,n):
    p=path_for(work,mechanism,n);meta=__import__('json').loads(p.with_suffix('.json').read_text())
    if file_hash(p)!=meta['sha256']:raise ValueError(f'Matrix hash changed: {p}')
    with np.load(p) as f:a={k:f[k].copy() for k in f.files}
    return dict(a,path=p,metadata=meta)
def prepare():
    rows=[]
    for n in (225,250,310):
        for work,mech in [('identity','iid')]+[(w,m) for w in ('sgd','momentum','momentum-bias') for m in ('bandinvmf','b2lt','dtc')]:
            p=path_for(work,mech,n);p.parent.mkdir(exist_ok=True,parents=True)
            if not p.exists():
                start=time.monotonic();A=workload(work,n)
                if mech=='iid':C=D=np.eye(n);d=np.ones(1);meta=dict(optimizer='none')
                else:C,D,d,meta=optimize(A,mech,n//5,work)
                audit=audit_arrays(A,C,D);meta.update(audit=audit,seconds=time.monotonic()-start,workload=work,mechanism=mech,total_steps=n)
                if mech=='bandinvmf':d=d[:4]
                elif mech=='b2lt':d=d[:n//5]
                np.savez(p,A=A,C=C,D=D,d=d,covariance_clip1=(audit['sigma_sum_clip1']**2)*(D@D.T))
                meta['sha256']=file_hash(p);save_json(p.with_suffix('.json'),meta)
            a=load(work,mech,n);audit=audit_arrays(a['A'],a['C'],a['D'])
            if not np.isclose(audit['objective'],a['metadata']['audit']['objective'],rtol=1e-10):raise ValueError('Matrix objective changed')
            rows.append(dict(workload=work,mechanism=mech,total_steps=n,**audit,solve_seconds=a['metadata'].get('seconds')))
            print(f"matrix {work}/{mech}/T{n}: E2={audit['objective']:.8g}",flush=True)
    from exp16a.support import write_csv
    write_csv(RESULTS/'matrix_comparison.csv',rows);save_json(RESULTS/'matrix_audit.json',rows)
    return rows
if __name__=='__main__':prepare()
