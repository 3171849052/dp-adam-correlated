"""First moments always consume already noised gradients in the clipping space."""
import math
import torch
from exp15d.config import METHODS


class Optimizer(torch.optim.Optimizer):
    def __init__(self, params, method, lr, betas=(.9, .999), eps_scale=.05):
        if method not in METHODS: raise ValueError(method)
        if not math.isfinite(lr) or lr <= 0: raise ValueError('Invalid lr')
        if not all(0 <= b < 1 for b in betas): raise ValueError('Invalid betas')
        if not math.isfinite(eps_scale) or eps_scale <= 0: raise ValueError('Invalid denominator constant')
        super().__init__(params, dict(method=method, lr=lr, betas=betas, eps_scale=eps_scale))
        self.last_diagnostics = {}

    @torch.no_grad()
    def freeze_scale(self):
        """Freeze d_(t-1) before all microbatches; vhat_0=0, d_0=eps_s."""
        for group in self.param_groups:
            b2 = group['betas'][1]
            for p in group['params']:
                state = self.state[p]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(p))
                    if group['method'].startswith('stpex'): state['exp_avg_sq'] = torch.zeros_like(p)
                if group['method'].startswith('stpex'):
                    vhat = state['exp_avg_sq']/(1-b2**state['step']) if state['step'] else torch.zeros_like(p)
                    denominator = vhat.sqrt()+group['eps_scale']
                else:
                    denominator = torch.ones_like(p)
                p._logical_denominator = denominator
                p._logical_scale = denominator.reciprocal()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None: raise ValueError('Pass already noised gradients; closures are unsupported')
        rows=[]; pending=[]
        # Validate every state/update before changing any parameter.
        for group in self.param_groups:
            method=group['method']; b1,b2=group['betas']
            for p in group['params']:
                if p.grad is None: raise RuntimeError('Missing noised gradient')
                if not hasattr(p, '_logical_denominator'): raise RuntimeError('freeze_scale must precede step')
                state=self.state[p]; t=state['step']+1; q=p.grad
                u=b1*state['exp_avg']+(1-b1)*q
                update=-group['lr']*(u if method in ('momentum','momentum+iid') else u/(1-b1**t))
                v=None
                if method.startswith('stpex'):
                    source={'stpex':lambda:p._logical_denominator*q,
                            'stpex2':lambda:q, 'stpex-adam3':lambda:u}[method]()
                    v=b2*state['exp_avg_sq']+(1-b2)*source.square()
                check=[q,u,update,p+update]+([v] if v is not None else [])
                if not all(torch.isfinite(x).all().item() for x in check):
                    raise FloatingPointError('Non-finite noised gradient, state, update or parameter')
                d=p._logical_denominator
                rows.append(torch.stack((q.double().square().sum(), update.double().square().sum(),
                    d.double().sum(),d.min().double(),d.max().double(),
                    torch.zeros((),device=p.device,dtype=torch.float64) if v is None else v.double().sum(),
                    torch.zeros((),device=p.device,dtype=torch.float64) if v is None else v.max().double(),
                    torch.tensor(p.numel(),device=p.device,dtype=torch.float64))))
                pending.append((p,state,t,u,v,update))
        r=torch.stack(rows).cpu(); n=float(r[:,7].sum())
        self.last_diagnostics=dict(step=pending[0][2],noisy_gradient_norm=float(r[:,0].sum().sqrt()),
            update_norm=float(r[:,1].sum().sqrt()),denominator_mean=float(r[:,2].sum()/n),
            denominator_min=float(r[:,3].min()),denominator_max=float(r[:,4].max()),
            v_mean=float(r[:,5].sum()/n),v_max=float(r[:,6].max()),finite=True)
        for p,state,t,u,v,update in pending:
            state['step']=t; state['exp_avg'].copy_(u)
            if v is not None: state['exp_avg_sq'].copy_(v)
            p.add_(update)

