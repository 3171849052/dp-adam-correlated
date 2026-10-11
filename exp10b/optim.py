"""Diagnostics are postprocessing of standard PyTorch Adam's DP state."""
import torch

class PreconditionerDiagnostics:
    def __init__(self): self.previous=None

    @torch.no_grad()
    def record(self,optimizer):
        current=[]
        for group in optimizer.param_groups:
            for p in group['params']:
                state=optimizer.state[p]; t=int(state['step'].item())
                den=state['exp_avg_sq'].flatten()[:64].sqrt()/((1-group['betas'][1]**t)**.5)+group['eps']
                current.append(1/den)
        cur=torch.cat(current)
        result=dict(preconditioner_mean=float(cur.mean()),
            preconditioner_std=float(cur.std(unbiased=False)),
            preconditioner_change_relative=0. if self.previous is None else
                float((cur-self.previous).norm()/self.previous.norm().clamp_min(1e-30)))
        self.previous=cur.clone()
        return result
