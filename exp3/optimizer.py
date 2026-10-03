"""Single-device hybrid Muon/Adam; the optimizer consumes one joint release."""
import re
import torch

MUON_PATTERN = re.compile(r'blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(0|2))\.weight$')


def partition(model):
    muon, adam = {}, {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            (muon if MUON_PATTERN.fullmatch(name) else adam)[name] = p
    assert muon and adam
    assert all(p.ndim == 2 for p in muon.values())
    a, b = {id(p) for p in muon.values()}, {id(p) for p in adam.values()}
    assert not a & b
    assert a | b == {id(p) for p in model.parameters() if p.requires_grad}
    return muon, adam


def muon_map(h):
    """Phi = Frobenius normalization followed by the standard quintic NS5.

    FP32 is used both for optimization and JVP diagnostics. Shape LR scaling
    belongs to the optimizer, not Phi. Coefficients follow KellerJordan/Muon.
    """
    assert h.ndim == 2
    x = h.mT if h.shape[0] > h.shape[1] else h
    x = x / (x.norm() + 1e-7)
    for _ in range(5):
        a = x @ x.mT
        b = -4.7750 * a + 2.0315 * (a @ a)
        x = 3.4445 * x + b @ x
    return x.mT if h.shape[0] > h.shape[1] else x


class HybridOptimizer(torch.optim.Optimizer):
    def __init__(self, model, muon_lr, adam_lr):
        self.muon, self.adam = partition(model)
        super().__init__([
            dict(params=list(self.muon.values()), lr=muon_lr, use_muon=True,
                 momentum=.95, nesterov=True, ns_steps=5, weight_decay=0.),
            dict(params=list(self.adam.values()), lr=adam_lr, use_muon=False,
                 betas=(.9, .999), eps=1e-8, weight_decay=0.)], {})
        self.step_count = 0

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                assert p.grad is not None
                s = self.state[p]
                g = p.grad
                if group['use_muon']:
                    if not s:
                        s['momentum_buffer'] = torch.zeros_like(p)
                    s['momentum_buffer'].lerp_(g, 1 - group['momentum'])
                    h = torch.lerp(g, s['momentum_buffer'], group['momentum'])
                    s['pre_ns'] = h.clone()  # actual completed Nesterov matrix
                    update = muon_map(h) * max(1., p.shape[0] / p.shape[1]) ** .5
                else:
                    if not s:
                        s.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                    s['step'] += 1
                    b1, b2 = group['betas']
                    s['exp_avg'].lerp_(g, 1 - b1)
                    s['exp_avg_sq'].lerp_(g.square(), 1 - b2)
                    m = s['exp_avg'] / (1 - b1 ** s['step'])
                    v = s['exp_avg_sq'] / (1 - b2 ** s['step'])
                    update = m / (v.sqrt() + group['eps'])
                p.add_(update, alpha=-group['lr'])
        self.step_count += 1
