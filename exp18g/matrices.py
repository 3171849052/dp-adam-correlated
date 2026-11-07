"""Audit the existing converged Exp18d SGD workload, never substitute a matrix."""
import numpy as np
import torch
from exp18g.common import ROOT,BASE,load_matrix,read,save,sha
from exp18g.privacy import fixed_epoch_sensitivity

def audit():
    torch.set_num_threads(2)
    m=load_matrix();meta=m['metadata']
    assert meta['workload']=='sgd' and meta['selected']=='b2lt'
    assert {k:meta[k] for k in ('T','K','P','Q')}==dict(T=310,K=62,P=62,Q=5)
    assert sha(ROOT/'exp18d/matrices/sgd.npz')==meta['sha256']
    assert meta['solver_success'] and meta['first_order_pass'] and meta['projected_gradient_inf']<=1e-7
    A,C,D=m['A'],m['C'],m['D']
    np.testing.assert_array_equal(A,np.tril(np.ones((310,310))))
    np.testing.assert_allclose(C@D,np.eye(310),atol=1e-8)
    np.testing.assert_allclose((A@D)@C,A,atol=1e-8)
    th=np.array(meta['theta']);om=np.array(meta['omega'])
    d=np.r_[1,(om[:,None]*th[:,None]**np.arange(61)).sum(0),np.zeros(248)]
    np.testing.assert_allclose(d,m['d'],atol=1e-14)
    sens=fixed_epoch_sensitivity(C,5,62)
    assert abs(sens-meta['audit']['sensitivity'])<1e-10
    # Reconstruct softmax coordinates and independently differentiate the selected
    # point of Exp18d's barrier objective (the solver also reports its endpoint).
    g=np.diff(np.r_[0,th,1]);h=np.r_[-om/(1-th),1+(om/(1-th)).sum()]
    assert (g>0).all() and (h>0).all()
    z=torch.tensor(np.r_[np.log(g[:5]/g[5]),np.log(h[:5]/h[5])],dtype=torch.float64,requires_grad=True)
    lag=torch.arange(310)[:,None]-torch.arange(310)[None,:]
    def objective(x):
        g=torch.softmax(torch.cat((x[:5],x.new_zeros(1))),0)
        h=torch.softmax(torch.cat((x[5:],x.new_zeros(1))),0)
        th=g[:5].cumsum(0);om=-(1-th)*h[:5]
        d=torch.cat((x.new_ones(1),(om[:,None]*th[:,None]**torch.arange(61)).sum(0),x.new_zeros(248)))
        D=torch.where(lag>=0,d[lag.clamp_min(0)],0.)
        C=torch.linalg.solve_triangular(D,torch.eye(310,dtype=torch.float64),upper=False)
        E=C[:,::62].sum(1).square().sum()*(torch.tensor(A)@D).square().sum()/310
        phi=-(-om).log().sum()-(th*(1-th)).log().sum()-(1+om.sum()).log()-(1+(om/(1-th)).sum()).log()-torch.pdist(th[:,None]).log().sum()
        return E,phi
    E0=float(objective(torch.tensor(np.r_[np.zeros(5),np.full(5,np.log(.2))]))[0])
    E,phi=objective(z);(E/E0+.001*phi).backward()
    pg=float((z-(z-z.grad).clamp(-12,12)).abs().max().detach())
    # Exp18d records convergence at the barrier-objective endpoint, while
    # selecting the lowest unpenalized E along the trace. They are distinct.
    endpoint=meta['trace'][-1]
    assert endpoint['gradient_inf']<=1e-7
    assert abs(endpoint['gradient_inf']-meta['projected_gradient_inf'])<1e-12
    assert endpoint['objective']==meta['endpoint_objective']
    assert abs(float(E.detach())-meta['selected_error'])<1e-9
    result=dict(passed=True,source='exp18d/matrices/sgd.npz',sha256=meta['sha256'],
        workload='sgd',T=310,K=62,P=62,Q=5,solver_success=meta['solver_success'],
        solver_message=meta['solver_message'],endpoint_projected_gradient_inf=meta['projected_gradient_inf'],
        selected_projected_gradient_inf=pg,first_order_pass=True,
        first_order_scope='Solver barrier-objective endpoint; selected matrix minimizes unpenalized E along trace',
        selected_is_endpoint=False,sensitivity=sens,
        inverse_error=float(abs(C@D-np.eye(310)).max()),objective=float(sens*sens*np.sum((A@D)**2)/310))
    save(BASE/'acceptance/matrix_audit.json',result)
    return result
