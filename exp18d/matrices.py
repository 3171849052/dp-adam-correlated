"""Exp18 section 5 solver, generalized only to T=310, k=5, K=P=62; epsilon=8."""
import numpy as np,torch
from scipy.optimize import minimize
from exp18d import BASE
RESULTS=BASE
from exp18.common import save,read,sha
from exp18.privacy import fixed_epoch_sensitivity,target_mu,epsilon_from_mu

def workload(name,n=310):
    t=np.arange(1,n+1);lag=t[:,None]-t[None,:]
    if name=='sgd':return np.tril(np.ones((n,n)))
    if name=='momentum':return np.where(lag>=0,1-.9**(np.maximum(lag,0)+1),0.)
    moments=np.where(lag>=0,.9**np.maximum(lag,0),0.)*(.1/(1-.9**t))[:,None]
    return np.tril(np.cumsum(moments,axis=0))

def audit(A,C,D):
    s=fixed_epoch_sensitivity(C,5,62);mu=target_mu(8,1e-5)
    assert np.max(np.abs(C@D-np.eye(310)))<1e-8
    assert np.max(np.abs((A@D)@C-A))<1e-8
    assert np.allclose(s,max(np.linalg.norm(C[:,j+62*np.arange(5)].sum(1)) for j in range(62)))
    eps=epsilon_from_mu(s/(s/mu),1e-5);assert abs(eps-8)<1e-9
    return dict(sensitivity=s,mu=mu,epsilon=eps,delta=1e-5,sigma_sum_clip1=s/mu,objective=float(s*s*np.sum((A@D)**2)/310),inverse_error=float(np.max(np.abs(C@D-np.eye(310)))))

def optimize(name):
    torch.set_num_threads(2);A=workload(name);W=torch.tensor(A,dtype=torch.float64);eye=torch.eye(310,dtype=torch.float64)
    lag=torch.arange(310)[:,None]-torch.arange(310)[None,:];records=[];best=[float('inf'),None];E0=[None]
    def build(z):
        g=torch.softmax(torch.cat((z[:5],z.new_zeros(1))),0);h=torch.softmax(torch.cat((z[5:],z.new_zeros(1))),0)
        th=g[:5].cumsum(0);om=-(1-th)*h[:5]
        d=torch.cat((z.new_ones(1),(om[:,None]*th[:,None]**torch.arange(61)).sum(0),z.new_zeros(248)))
        D=torch.where(lag>=0,d[lag.clamp_min(0)],0.);C=torch.linalg.solve_triangular(D,eye,upper=False)
        sens2=C[:,::62].sum(1).square().sum();E=sens2*(W@D).square().sum()/310
        phi=-(-om).log().sum()-(th*(1-th)).log().sum()-(1+om.sum()).log()-(1+(om/(1-th)).sum()).log()-torch.pdist(th[:,None]).log().sum()
        return E,phi,C,D,th,om
    z0=np.r_[np.zeros(5),np.full(5,np.log(.2))];E0[0]=float(build(torch.tensor(z0))[0])
    def fun(z):
        x=torch.tensor(z,dtype=torch.float64,requires_grad=True);E,phi,*_=build(x);f=E/E0[0]+.001*phi;f.backward();grad=x.grad.numpy().copy()
        records.append(dict(objective=float(f.detach()),error=float(E.detach()),gradient_l2=float(np.linalg.norm(grad)),gradient_inf=float(abs(grad).max())))
        if float(E.detach())<best[0]:best[:]=[float(E.detach()),z.copy()]
        return float(f.detach()),grad
    result=minimize(fun,z0,jac=True,method='L-BFGS-B',bounds=[(-12,12)]*10,options=dict(maxiter=200,ftol=1e-14,gtol=1e-9,maxls=50))
    f,grad=fun(result.x);pg=float(np.max(abs(result.x-np.clip(result.x-grad,-12,12))))
    E,phi,C,D,th,om=build(torch.tensor(best[1]));C=C.detach().numpy();D=D.detach().numpy()
    iid=5*np.sum(A*A)/310;selected='b2lt'
    if best[0]>iid:C=D=np.eye(310);selected='iid'
    meta=dict(workload=name,T=310,K=62,P=62 if selected=='b2lt' else 1,Q=5,theta=th.tolist(),omega=om.tolist(),selected=selected,solver_success=bool(result.success),solver_message=str(result.message),iterations=int(result.nit),projected_gradient_inf=pg,first_order_pass=pg<=1e-7,endpoint_objective=f,selected_error=min(best[0],iid),iid_error=float(iid),trace=records,audit=audit(A,C,D))
    # Endpoint convergence and selected mechanism are distinct PDF requirements.
    if pg>1e-7:raise RuntimeError(f'{name}: projected gradient {pg} exceeds PDF tolerance')
    p=RESULTS/'matrices'/f'{name}.npz';p.parent.mkdir(parents=True,exist_ok=True)
    np.savez(p,A=A,C=C,D=D,d=D[:,0]);meta['sha256']=sha(p);save(p.with_suffix('.json'),meta);print(name,meta['audit'],flush=True)

def load(name):
    p=RESULTS/'matrices'/f'{name}.npz';meta=read(p.with_suffix('.json'));assert sha(p)==meta['sha256']
    with np.load(p) as f:a={k:f[k].copy() for k in f.files}
    return dict(a,metadata=meta)
if __name__=='__main__':
    for name in ('sgd','momentum','momentum-bias'):optimize(name)
