"""stp-ex2.pdf sections 2/6: adaptive scalar alpha, d0=1, 24 bisections."""
import torch

def adaptive(q,d,vhat):
    M=vhat.sum();c=d.max().clamp_min(1);ratio=d/c
    def energy(a):return (c*ratio.pow(a)*q).square().sum()
    if energy(1)>=M:a=1.
    elif energy(0)<M:a=0.
    else:
        low,high=0.,1.
        for _ in range(24):
            mid=(low+high)/2
            if energy(mid)>=M:low=mid
            else:high=mid
        a=max(0.,low-1e-6)
    return a,c*ratio.pow(a)*q

class Update:
    def __init__(self,model,group,lr,eps):
        self.ps=list(model.parameters());self.group=group;self.lr=lr;self.eps=eps;self.t=0
        flat=torch.cat([p.detach().flatten() for p in self.ps]);self.u=torch.zeros_like(flat);self.v=torch.zeros_like(flat);self.d=torch.ones_like(flat)
    @torch.no_grad()
    def step(self,q):
        self.t+=1;t=self.t;alpha=None
        self.u.mul_(.9).add_(q,alpha=.1)
        if self.group=='stpex2':
            old=self.v/(1-.999**(t-1)) if t>1 else self.v
            alpha,r=adaptive(q,self.d,old);self.v.mul_(.999).addcmul_(r,r,value=.001)
            self.d=(self.v/(1-.999**t)).sqrt()+self.eps;update=self.u/(1-.9**t)
        elif self.group=='adam':
            self.v.mul_(.999).addcmul_(q,q,value=.001)
            update=(self.u/(1-.9**t))/((self.v/(1-.999**t)).sqrt()+1e-8)
        else:update=self.u
        if not torch.isfinite(update).all() or not torch.isfinite(self.v).all():raise FloatingPointError('nonfinite optimizer')
        offset=0
        for p in self.ps:
            p.add_(update[offset:offset+p.numel()].view_as(p),alpha=-self.lr);offset+=p.numel()
        return alpha
    def state_dict(self):return {k:getattr(self,k) for k in ('u','v','d','t','group','lr','eps')}
