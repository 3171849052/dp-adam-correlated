"""Freeze only Adam's update denominator after the block's first v update."""
import torch
class BlockAdam(torch.optim.Optimizer):
    def __init__(self,params,lr,block_size):
        assert block_size>=1 and int(block_size)==block_size
        super().__init__(params,dict(lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.,block_size=int(block_size)))
    @torch.no_grad()
    def step(self,closure=None):
        assert closure is None
        for group in self.param_groups:
            beta1,beta2=group['betas']
            for p in group['params']:
                assert p.grad is not None
                state=self.state[p]
                if not state:
                    state.update(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
                state['step'].add_(1);t=int(state['step'].item())
                m,v=state['exp_avg'],state['exp_avg_sq'];g=p.grad
                m.lerp_(g,1-beta1);v.mul_(beta2).addcmul_(g,g,value=1-beta2)
                if (t-1)%group['block_size']==0:
                    state['block_denominator']=v.sqrt().div_((1-beta2**t)**.5).add_(group['eps'])
                    state['block_start']=t
                p.addcdiv_(m,state['block_denominator'],value=-group['lr']/(1-beta1**t))
class PreconditionerDiagnostics:
    """Postprocessing of DP Adam state; no unclipped/private training statistics."""
    def __init__(self): self.previous=None
    @torch.no_grad()
    def record(self,optimizer):
        # Fixed coordinates bound diagnostic memory and are identical across runs.
        current=[];live=[];ages=[]
        for group in optimizer.param_groups:
            for p in group['params']:
                s=optimizer.state[p]; t=int(s['step'].item())
                den=s['exp_avg_sq'].flatten()[:64].sqrt()/((1-group['betas'][1]**t)**.5)+group['eps']
                active=s['block_denominator'].flatten()[:64] if 'block_denominator' in s else den
                current.append(1/active);live.append(1/den);ages.append(t-s.get('block_start',t))
        cur=torch.cat(current);reference=torch.cat(live)
        result=dict(preconditioner_mean=float(cur.mean()),preconditioner_change_relative=0. if self.previous is None else float((cur-self.previous).norm()/self.previous.norm().clamp_min(1e-30)),
            denominator_lag_relative=float(((reference/cur)-1).abs().mean()),block_age=max(ages))
        self.previous=cur.clone();return result
