import copy
import io
import json
import numpy as np
import torch
from exp14a import ROOT, RESULTS
from exp14a.optimizer import AdaM3
from exp14a.support import jobs, reference_dir, historical
from exp14a.bandinvmf import build_matrices
from exp14a.privacy import calibrate
from exp14a.noise import BandInvMFNoise
from exp14a.config import save_json, array_hash

def main():
    p=torch.nn.Parameter(torch.tensor([1.,-2.,3.],dtype=torch.float64))
    opt=AdaM3([p],lr=.003)
    ref=p.detach().clone();m=torch.zeros_like(p);v=torch.zeros_like(p)
    grads=[torch.tensor([.2*t,-.1,(-1.)**t],dtype=torch.float64) for t in range(1,31)]
    resumed=None
    for t,g in enumerate(grads,1):
        p.grad=g.clone();opt.step()
        m=.9*m+.1*g;v=.999*v+.001*m.square()
        ref=ref-.003*(m/(1-.9**t))/(torch.sqrt(v/(1-.999**t))+1e-8)
        torch.testing.assert_close(p,ref,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(opt.state[p]['exp_avg_sq'],v,rtol=1e-12,atol=1e-12)
        if resumed:
            q,other=resumed;q.grad=g.clone();other.step()
            torch.testing.assert_close(q,p,rtol=0,atol=0)
        if t==10:
            buf=io.BytesIO();torch.save(opt.state_dict(),buf);buf.seek(0)
            q=torch.nn.Parameter(p.detach().clone());other=AdaM3([q],lr=.003)
            other.load_state_dict(torch.load(buf,weights_only=True));resumed=(q,other)
    # Execute the original pure noise module without importing Exp9's write-producing initializer.
    original={};exec(compile((ROOT/'exp9/noise.py').read_text(),'exp9/noise.py','exec'),original)
    checks=[]
    for cfg in jobs():
        directory=reference_dir(cfg);summary=json.loads((directory/'summary.json').read_text())
        assert summary['status']=='completed'
        d,S,W=build_matrices(cfg.noise,cfg.total_steps,4,.9)
        with np.load(directory/'matrices.npz') as saved:
            for key,value in [('coefficients',d),('strategy',S),('workload',W)]:
                np.testing.assert_array_equal(value,saved[key])
        cal=calibrate(S,cfg.protocol())
        for k,value in cal.items(): assert summary['calibration'][k]==value,(cfg.task,cfg.method,k)
        params=[torch.zeros(3,4),torch.zeros(5)]
        a=BandInvMFNoise(params,d,cal['innovation_std_sum'],cfg.total_steps,cfg.seed+1)
        b=original['BandInvMFNoise'](params,d,cal['innovation_std_sum'],cfg.total_steps,cfg.seed+1)
        for _ in range(cfg.total_steps):
            for x,y in zip(a.next(),b.next()): torch.testing.assert_close(x,y,rtol=0,atol=0)
        if cfg.task=='cv': indices=np.arange(50000)
        else:
            with np.load(historical/'nlp_split.npz') as f: indices=f['train']
        order=indices[torch.randperm(len(indices),generator=torch.Generator().manual_seed(cfg.seed)).numpy()]
        np.testing.assert_array_equal(order,np.load(directory/'train_order.npy'))
        checks.append(dict(task=cfg.task,method=cfg.method,steps=cfg.total_steps,matrix_exact=True,calibration_exact=True,noise_stream_exact=True,order_exact=True))
        print(cfg.task,cfg.method,'PASS',flush=True)
    save_json(RESULTS/'tests.json',dict(status='passed',optimizer_float64_reference=True,state_save_restore_exact=True,checks=checks))

if __name__=='__main__': main()
