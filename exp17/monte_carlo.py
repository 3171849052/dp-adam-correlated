"""Fixed clipped paths; measured bias/variance/MSE, never asserted improvements.

Antithetic samples and zero-mean first/second Hermite control variates reduce
Monte Carlo error in small biases. Raw empirical variance/MSE remain unadjusted.
Smoothing is held fixed for the asymptotic test; r=lambda2/nu2 varies accordingly.
"""
import numpy as np
import torch
from exp17.optimizer import curvature
from exp17.common import save, RESULTS


def run(samples=200000):
    torch.set_num_threads(2)
    dtype=torch.float64;t=4;lambda2=.4
    path=torch.tensor([.6,.9,.7,1.],dtype=dtype)
    powers=torch.arange(t-1,-1,-1,dtype=dtype)
    a=.1*.9**powers/(1-.9**t);b=.001*.999**powers/(1-.999**t)
    noise=torch.randn(samples//2,t,generator=torch.Generator().manual_seed(776),dtype=dtype)
    rows=[]
    for method in ('lc','hc'):
        active=torch.zeros(t,dtype=dtype);active[-1]=1
        if method=='hc': active.fill_(1)
        def raw(x):
            M=(x*a).sum(-1);V=(x.square()*b).sum(-1)
            return M/((V+lambda2).sqrt()+1e-8)
        target=raw(path).item()
        hess=torch.autograd.functional.hessian(raw,path)
        for sd in (.16,.08,.04,.02):
            z=noise*active
            xplus=path+sd*z;xminus=path-sd*z
            vals=[]
            for x in (xplus,xminus):
                M=(a*x).sum(-1);V=(b*x*x).sum(-1)
                R=(x*(.9*.999)**powers).sum(-1);H=(x*x*(.999**2)**powers).sum(-1)
                rawv,corr=curvature(method,M,V,x[:,-1],t,lambda2/sd**2,sd**2,R,H)
                vals.append(torch.stack((rawv,rawv-corr)))
            both=torch.cat(vals,dim=1)
            paired=(vals[0]+vals[1])/2
            quadratic=.5*sd**2*(torch.einsum('ni,ij,nj->n',z,hess,z)-(hess.diag()*active).sum())
            adjusted=paired-quadratic[None,:]
            for i,name in enumerate(('raw','corrected')):
                v=both[i];bias=(adjusted[i]-target).mean().item()
                rows.append(dict(method=method,estimator=name,sd=sd,lambda2=lambda2,
                    bias=bias,bias_standard_error=adjusted[i].std().item()/len(noise)**.5,
                    empirical_bias=v.mean().item()-target,variance=v.var(unbiased=False).item(),
                    mse=((v-target)**2).mean().item(),samples=samples))
    slopes=[]
    for method in ('lc','hc'):
        for name in ('raw','corrected'):
            selected=[r for r in rows if r['method']==method and r['estimator']==name]
            slope=float(np.polyfit(np.log([r['sd'] for r in selected]),np.log([max(abs(r['bias']),1e-30) for r in selected]),1)[0])
            slopes.append(dict(method=method,estimator=name,measured_bias_log_slope=slope,
                expected_local_order=2 if name=='raw' else 4,
                resolved_at_smallest_noise=abs(selected[-1]['bias'])>3*selected[-1]['bias_standard_error']))
    result=dict(status='measured',rows=rows,scaling=slopes,fixed_path=path.tolist(),
                assumptions='Fixed clipped path and fixed lambda2>0. LC fixes past observations. '
                'No claim for adaptive training or uniform lambda2=r*nu2 limit. '
                'Noise pairs are independent; antithetic/control variates used only for bias estimates.')
    save(RESULTS/'monte_carlo.json',result)
    return result

if __name__=='__main__': run()
