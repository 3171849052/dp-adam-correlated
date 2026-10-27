"""The four specified coordinatewise updates; no probes or Hessians in training."""
import torch


def smooth(V, lambda2):
    rV = (V + lambda2).sqrt()
    h = 1 / (rV + 1e-8)
    hp = -1 / (2 * rV * (rV + 1e-8).square())
    hpp = (3*rV + 1e-8) / (4*rV.pow(3)*(rV + 1e-8).pow(3))
    return h, hp, hpp


def curvature(method, M, V, x, t, r, nu2, R=None, H=None):
    h, hp, hpp = smooth(V, r*nu2)
    if method == 'lc':
        b, d = .1/(1-.9**t), .001/(1-.999**t)
        lap = 2*d*M*hp + 4*b*d*x*hp + 4*d*d*x.square()*M*hpp
    else:
        S1 = (.1*.001/((1-.9**t)*(1-.999**t)))*R
        S2 = (.001**2/(1-.999**t)**2)*H
        lap = 2*M*hp + 4*S1*hp + 4*M*S2*hpp
    return M*h, (nu2/2)*lap


class PrivateAdam(torch.optim.Optimizer):
    def __init__(self, params, method, lr, nu2, r=None, correction=True):
        if method not in ('adam','bc','lc','hc'): raise ValueError(method)
        if nu2 <= 0 or (method != 'adam' and (r is None or r <= 0)):
            raise ValueError('Positive noise variance and smoothing/floor required')
        super().__init__(params, dict(lr=lr, method=method, nu2=nu2, r=r, correction=correction))
        self.diagnostics = {}

    @torch.no_grad()
    def step(self):
        # Accumulate diagnostics in FP64 without transferring whole tensors to CPU.
        first = self.param_groups[0]['params'][0]
        totals = torch.zeros(4, device=first.device, dtype=torch.float64)
        finite = torch.ones((), device=first.device, dtype=torch.bool)
        for group in self.param_groups:
            method, nu2, r = group['method'], group['nu2'], group['r']
            for p in group['params']:
                x = p.grad
                if x is None: raise ValueError('Every parameter must have a private gradient')
                state = self.state[p]
                if not state:
                    state.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                    if method == 'hc': state.update(R=torch.zeros_like(p), H=torch.zeros_like(p))
                state['t'] += 1
                t = state['t']
                m, v = state['m'], state['v']
                m.mul_(.9).add_(x, alpha=.1)
                v.mul_(.999).addcmul_(x,x,value=.001)
                M, V = m/(1-.9**t), v/(1-.999**t)
                if method == 'hc':
                    state['R'].mul_(.9*.999).add_(x)
                    state['H'].mul_(.999**2).addcmul_(x,x)
                if method == 'adam':
                    base = M/(V.sqrt()+1e-8); correction = torch.zeros_like(base)
                elif method == 'bc':
                    base = M/(V-nu2).clamp_min(r*nu2).sqrt(); correction = torch.zeros_like(base)
                else:
                    base, correction = curvature(method,M,V,x,t,r,nu2,state.get('R'),state.get('H'))
                direction = base-correction if group['correction'] else base
                finite &= torch.isfinite(direction).all()
                totals[0] += correction.double().square().sum()
                totals[1] += direction.double().square().sum()
                totals[2] += base.double().square().sum()
                totals[3] += (direction.double()*base.double()).sum()
                p.add_(direction,alpha=-group['lr'])
                finite &= torch.isfinite(p).all()
        if not finite.item(): raise FloatingPointError('Non-finite direction or parameter')
        c,d,b,dot = totals.cpu().tolist()
        self.diagnostics = dict(correction_norm=c**.5, direction_norm=d**.5,
            update_norm=group['lr']*d**.5, base_norm=b**.5,
            direction_cosine=dot/(d*b)**.5 if d*b>0 else 1., finite=True)
