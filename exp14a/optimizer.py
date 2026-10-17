"""AdaM³: Adam with uncorrected first-moment square as the v input."""
import json
import torch

class AdaM3(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(.9,.999), eps=1e-8, weight_decay=0., diagnostics_path=None):
        assert weight_decay == 0.
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))
        self.diagnostics_path = diagnostics_path

    @torch.no_grad()
    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        stats=[]; anomalies=0; record=False; t=0
        for group in self.param_groups:
            b1,b2=group['betas']
            for p in group['params']:
                if p.grad is None: continue
                state=self.state[p]
                if not state:
                    state.update(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
                state['step'].add_(1);t=int(state['step'].item())
                m,v=state['exp_avg'],state['exp_avg_sq']
                m.mul_(b1).add_(p.grad,alpha=1-b1)
                v.mul_(b2).addcmul_(m,m,value=1-b2)
                vhat=v/(1-b2**t)
                update=(m/(1-b1**t))/(vhat.sqrt()+group['eps'])
                update.mul_(-group['lr'])
                record=t<=20 or t%10==0
                finite=torch.isfinite(update).all() & torch.isfinite(v).all() & torch.isfinite(m).all()
                anomalies+=int(not finite.item())
                if record:
                    stats.append(torch.stack((update.double().square().sum(),v.double().sum(),v.min().double(),v.max().double(),vhat.double().sum(),vhat.max().double(),torch.tensor(p.numel(),device=p.device,dtype=torch.float64))))
                p.add_(update)
        if self.diagnostics_path is not None and (record or anomalies):
            row=dict(step=t,nonfinite_parameter_tensors=anomalies)
            if stats:
                s=torch.stack(stats).cpu();n=float(s[:,6].sum())
                row.update(update_norm=float(s[:,0].sum().sqrt()),v_mean=float(s[:,1].sum()/n),v_min=float(s[:,2].min()),v_max=float(s[:,3].max()),vhat_mean=float(s[:,4].sum()/n),vhat_max=float(s[:,5].max()))
            with self.diagnostics_path.open('a') as f: f.write(json.dumps(row,allow_nan=True)+'\n')
        return loss
