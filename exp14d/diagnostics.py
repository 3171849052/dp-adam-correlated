"""Read-only optimizer hooks; preserve torch.optim.Adam update implementation."""
import json
import torch

def attach(optimizer,path):
    state={'step':0}
    def before(opt,args,kwargs):
        state['step']+=1
        state['record']=state['step']<=20 or state['step']%10==0
        if state['record']:
            state['before']=[p.detach().clone() for g in opt.param_groups for p in g['params'] if p.grad is not None]
    @torch.no_grad()
    def after(opt,args,kwargs):
        if not state['record']: return
        params=[p for g in opt.param_groups for p in g['params'] if p.grad is not None]
        stats=[]
        for p,old in zip(params,state.pop('before')):
            s=opt.state[p];v=s['exp_avg_sq'];vh=v/(1-.999**int(s['step'].item()))
            stats.append(torch.stack(((p.double()-old.double()).square().sum(),v.double().sum(),v.min().double(),v.max().double(),vh.double().sum(),vh.max().double(),torch.tensor(p.numel(),device=p.device,dtype=torch.float64))))
        x=torch.stack(stats).cpu();n=float(x[:,6].sum())
        row=dict(step=state['step'],lr=opt.param_groups[0]['lr'],update_norm=float(x[:,0].sum().sqrt()),v_mean=float(x[:,1].sum()/n),v_min=float(x[:,2].min()),v_max=float(x[:,3].max()),vhat_mean=float(x[:,4].sum()/n),vhat_max=float(x[:,5].max()))
        with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    optimizer.register_step_pre_hook(before);optimizer.register_step_post_hook(after)
