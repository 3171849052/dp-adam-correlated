import json
from dataclasses import replace
import numpy as np
import torch
from exp14b import ROOT, RESULTS
from exp14b.config import save_json, file_hash
from exp14b.support import jobs, reference_dir, frozen_row, BASE_LR, code_hashes
from exp14b.optimizer import AdaM3
from exp14b.bandinvmf import build_matrices
from exp14b.privacy import calibrate
from exp14b.noise import BandInvMFNoise
from exp14b.pipeline import rank

def main():
    grid=jobs();assert len(grid)==len({c.id for c in grid})==36
    for eps in (1e-16,1e-12,1e-8):
        p=torch.nn.Parameter(torch.tensor([1.,-2.,3.],dtype=torch.float64));ref=p.detach().clone()
        m=torch.zeros_like(p);v=torch.zeros_like(p);opt=AdaM3([p],lr=.00015,eps=eps)
        for t in range(1,31):
            g=torch.tensor([1e-14*t,-.1,(-1.)**t],dtype=torch.float64)
            p.grad=g;opt.step();m=.9*m+.1*g;v=.999*v+.001*m.square()
            ref=ref-.00015*(m/(1-.9**t))/(torch.sqrt(v/(1-.999**t))+eps)
            torch.testing.assert_close(p,ref,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(opt.state[p]['exp_avg_sq'],v,rtol=1e-12,atol=1e-12)
    tied=[dict(accuracy=.5,epochs=[dict(validation_loss=l)],number=n) for l,n in ((2,1),(1,3),(1,2))]
    assert [r['number'] for r in rank(tied)]==[2,3,1]
    with np.load(ROOT/'exp9/results/cv_split.npz') as split:
        a,b=split['train'],split['search_validation']
        assert len(a)==45000 and len(b)==5000 and len(set(a)&set(b))==0
    original={};exec(compile((ROOT/'exp9/noise.py').read_text(),'exp9/noise.py','exec'),original)
    checks=[]
    for base in grid[::9]:
        frozen=frozen_row('cv',base.method)
        assert float(frozen['C'])==30 and float(frozen['lr'])==BASE_LR[base.method]
        for stage in ('search','final'):
            cfg=replace(base,stage=stage);directory=reference_dir(cfg)
            ref=json.loads((directory/'summary.json').read_text())
            d,S,W=build_matrices(cfg.noise,cfg.total_steps,4,.9)
            with np.load(directory/'matrices.npz') as saved:
                for k,value in [('coefficients',d),('strategy',S),('workload',W)]: np.testing.assert_array_equal(value,saved[k])
            cal=calibrate(S,cfg.protocol())
            # Historical search may use another C; normalize its innovation scale.
            assert cal['sensitivity']==ref['calibration']['sensitivity']
            assert np.isclose(cal['innovation_std_sum']/cfg.C,ref['calibration']['innovation_std_sum']/ref['C'])
            params=[torch.zeros(3,4),torch.zeros(5)]
            x=BandInvMFNoise(params,d,cal['innovation_std_sum'],cfg.total_steps,cfg.seed+1)
            y=original['BandInvMFNoise'](params,d,cal['innovation_std_sum'],cfg.total_steps,cfg.seed+1)
            for _ in range(cfg.total_steps):
                for u,v in zip(x.next(),y.next()): torch.testing.assert_close(u,v,rtol=0,atol=0)
            checks.append(dict(method=cfg.method,stage=stage,steps=cfg.total_steps,reference=str(directory),matrix_exact=True,calibration=True,noise_exact=True))
            print(cfg.method,stage,'PASS',flush=True)
    from exp14b.cv_model import checkpoint_path, checkpoint_sha256
    path=checkpoint_path()
    assert (ROOT/'data/cifar-100-python/train').is_file() and (ROOT/'data/cifar-100-python/test').is_file()
    save_json(RESULTS/'search_grid.json',[c.protocol()|dict(trial_id=c.id) for c in grid])
    save_json(RESULTS/'source_provenance.json',code_hashes())
    save_json(RESULTS/'tests.json',dict(status='passed',optimizer_all_eps_float64=True,selection_tiebreak=True,split=[45000,5000],checkpoint=str(path),checkpoint_sha256=checkpoint_sha256(path),checks=checks))
if __name__=='__main__': main()
