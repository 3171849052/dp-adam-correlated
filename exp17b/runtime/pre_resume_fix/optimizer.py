"""Same Adam state for all methods; deterministic corrections use frozen history."""
import math
import torch


def scalar_f(u, A, B, K, D, r, nu2):
    return (A+B*u)/((K+D*u.square()+r*nu2).sqrt()+1e-8)


def second_derivative(M, V, x, t, r, nu2):
    z=(V+r*nu2).sqrt()
    hp=-1/(2*z*(z+1e-8).square())
    hpp=(3*z+1e-8)/(4*z.pow(3)*(z+1e-8).pow(3))
    b,d=.1/(1-.9**t),.001/(1-.999**t)
    return 2*d*M*hp+4*b*d*x*hp+4*d*d*x.square()*M*hpp


def directions(method, x, m, v, t, r, nu2, alpha):
    """m,v are PRE-step states. Returns base, subtracted correction, new m,v."""
    B,D=.1/(1-.9**t),.001/(1-.999**t)
    new_m=.9*m+.1*x
    new_v=.999*v+.001*x.square()
    M,V=new_m/(1-.9**t),new_v/(1-.999**t)
    base=M/((V+r*nu2).sqrt()+1e-8)
    if method=='smooth' or alpha==0:
        correction=torch.zeros_like(base)
    elif method=='lc':
        correction=(alpha*nu2/2)*second_derivative(M,V,x,t,r,nu2)
    elif method=='gh':
        A,K=.9*m/(1-.9**t),.999*v/(1-.999**t)
        c=math.sqrt(3*nu2)
        correction=(alpha/6)*(scalar_f(x+c,A,B,K,D,r,nu2)+scalar_f(x-c,A,B,K,D,r,nu2)-2*base)
    else:
        raise ValueError(method)
    return base,correction,new_m,new_v


class PrivateAdam(torch.optim.Optimizer):
    def __init__(self, params, method, lr, nu2, r, alpha=0):
        if method not in ('smooth','lc','gh') or r<0 or nu2<=0 or not 0<=alpha<=1:
            raise ValueError('Invalid method, smoothing, noise or alpha')
        if method=='lc' and alpha and r==0:
            raise ValueError('LC requires positive smoothing in this experiment')
        super().__init__(params,dict(lr=lr,method=method,nu2=nu2,r=r,alpha=alpha))
        self.diagnostics={}

    @torch.no_grad()
    def step(self):
        totals=torch.zeros(5,device=self.param_groups[0]['params'][0].device,dtype=torch.float64)
        finite=torch.ones((),device=totals.device,dtype=torch.bool)
        for group in self.param_groups:
            for p in group['params']:
                x=p.grad
                if x is None: raise ValueError('Private gradient missing')
                state=self.state[p]
                if not state: state.update(t=0,m=torch.zeros_like(p),v=torch.zeros_like(p))
                state['t']+=1
                base,c,m,v=directions(group['method'],x,state['m'],state['v'],state['t'],group['r'],group['nu2'],group['alpha'])
                state['m'].copy_(m);state['v'].copy_(v)
                direction=base-c
                finite &= torch.isfinite(direction).all()
                bd,cd,dd=base.double(),c.double(),direction.double()
                totals+=torch.stack([cd.square().sum(),dd.square().sum(),bd.square().sum(),(dd*bd).sum(),(cd*bd).sum()])
                p.add_(direction,alpha=-group['lr'])
                finite &= torch.isfinite(p).all()
        if not finite.item(): raise FloatingPointError('Non-finite direction or parameter')
        c,d,b,dot,cb=totals.cpu().tolist()
        parallel=cb/b**.5 if b else 0.
        self.diagnostics=dict(correction_norm=c**.5,direction_norm=d**.5,update_norm=group['lr']*d**.5,
            base_norm=b**.5,direction_cosine=dot/(d*b)**.5 if d*b else 1.,
            relative_update_scale=(d/b)**.5 if b else 0.,correction_parallel=parallel,
            correction_orthogonal=max(0.,c-parallel**2)**.5,finite=True)
