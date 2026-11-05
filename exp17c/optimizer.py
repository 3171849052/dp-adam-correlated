"""Unsmooth LC, frozen FP32 Adam history, analytic FP64 curvature."""
import torch

def second_derivative(x,m,v,t):
    u=x.double()
    if t==1:
        return -2e-8*u.sign()/(u.abs()+1e-8).pow(3)
    B,D=.1/(1-.9**t),.001/(1-.999**t)
    A,K=.9*m.double()/(1-.9**t),.999*v.double()/(1-.999**t)
    M=A+B*u
    z=(K+D*u.square()).sqrt()
    # At zero history and zero input curvature is undefined; correction is zero.
    mask=z>0
    out=torch.zeros_like(u)
    zz=z[mask];uu=u[mask];mm=M[mask]
    hp=-1/(2*zz*(zz+1e-8).square())
    hpp=(3*zz+1e-8)/(4*zz.pow(3)*(zz+1e-8).pow(3))
    out[mask]=2*D*mm*hp+4*B*D*uu*hp+4*D*D*uu.square()*mm*hpp
    return out

def directions(method,x,m,v,t,r,nu2,alpha):
    assert r==0
    nm=.9*m+.1*x;nv=.999*v+.001*x.square()
    base=(nm/(1-.9**t))/((nv/(1-.999**t)).sqrt()+1e-8)
    c=second_derivative(x,m,v,t)*(alpha*nu2/2) if alpha else torch.zeros_like(x,dtype=torch.float64)
    return base,c,nm,nv

class PrivateAdam(torch.optim.Optimizer):
    def __init__(self,params,method,lr,nu2,r,alpha=0):
        if r!=0 or alpha<0 or nu2<=0: raise ValueError('Require r=0, alpha>=0, nu2>0')
        super().__init__(params,dict(method=method,lr=lr,nu2=nu2,r=r,alpha=alpha))
        self.diagnostics={}
    @torch.no_grad()
    def step(self):
        bases=[];corrections=[];inputs=[]
        for g in self.param_groups:
            for p in g['params']:
                x=p.grad
                state=self.state[p]
                if not state: state.update(t=0,m=torch.zeros_like(p),v=torch.zeros_like(p))
                state['t']+=1
                b,c,m,v=directions(g['method'],x,state['m'],state['v'],state['t'],0,g['nu2'],g['alpha'])
                state['m'].copy_(m);state['v'].copy_(v)
                d=b.double()-c
                if not torch.isfinite(d).all(): raise FloatingPointError('Nonfinite LC direction')
                p.add_(d.to(p.dtype),alpha=-g['lr'])
                if not torch.isfinite(p).all(): raise FloatingPointError('Nonfinite parameter')
                bases.append(b.flatten().double());corrections.append(c.flatten());inputs.append(x.flatten())
        b=torch.cat(bases);c=torch.cat(corrections);x=torch.cat(inputs);d=b-c
        bn=b.norm().item();cn=c.norm().item();dn=d.norm().item()
        quant=torch.quantile(c.abs(),torch.tensor([.5,.9,.99,.999],device=c.device,dtype=c.dtype)).tolist()
        self.diagnostics=dict(base_norm=bn,correction_norm=cn,direction_norm=dn,
            base_update_norm=g['lr']*bn,update_norm=g['lr']*dn,
            correction_max_abs=c.abs().max().item(),
            **dict(zip(['correction_abs_p50','correction_abs_p90','correction_abs_p99','correction_abs_p999'],quant)),
            correction_base_cosine=(c@b).item()/(cn*bn) if cn*bn else 0.,
            input_min_abs=x.abs().min().item(),input_zero_count=(x==0).sum().item(),
            first_step=state['t']==1,relative_update_scale=dn/bn if bn else 0.,finite=True)
