"""Numerical optimizer, exact historical matrices/RNG, splits and calibration."""
import io
import json
from dataclasses import replace
import numpy as np
import torch
from exp14c import ROOT, RESULTS
from exp14c.config import Trial, METHODS, FINAL_SEEDS, save_json, file_hash
from exp14c.support import grid, reference_dir, historical, protected_snapshot, code_hashes
from exp14c.optimizer import AdaM3
from exp14c.noise import BandInvMFNoise
from exp14c.bandinvmf import build_matrices
from exp14c.privacy import calibrate, epsilon_from_mu

def main():
    save_json(RESULTS/'protected_before.json',protected_snapshot())
    p=torch.nn.Parameter(torch.tensor([1.,-2.,3.],dtype=torch.float64));opt=AdaM3([p],lr=.003)
    ref=p.detach().clone();m=torch.zeros_like(p);v=torch.zeros_like(p);resumed=None
    for t in range(1,31):
        g=torch.tensor([.2*t,-.1,(-1.)**t],dtype=torch.float64)
        p.grad=g.clone();opt.step();m=.9*m+.1*g;v=.999*v+.001*m.square()
        ref-=.003*(m/(1-.9**t))/(torch.sqrt(v/(1-.999**t))+1e-8)
        torch.testing.assert_close(p,ref,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(opt.state[p]['exp_avg_sq'],v,rtol=1e-12,atol=1e-12)
        if resumed:
            q,other=resumed;q.grad=g.clone();other.step();torch.testing.assert_close(q,p,rtol=0,atol=0)
        if t==10:
            buf=io.BytesIO();torch.save(opt.state_dict(),buf);buf.seek(0)
            q=torch.nn.Parameter(p.detach().clone());other=AdaM3([q],lr=.003)
            other.load_state_dict(torch.load(buf,weights_only=True));resumed=(q,other)
    original={};exec(compile((ROOT/'exp9/noise.py').read_text(),'exp9/noise.py','exec'),original)
    assets=json.loads((historical/'assets_manifest.json').read_text())
    for task,counts in [('cv',(45000,5000)),('nlp',(62000,5349))]:
        path=historical/f'{task}_split.npz';assert file_hash(path)==assets['splits'][task]['file_sha256']
        with np.load(path) as s:
            a,b=s['train'],s['search_validation'];assert (len(a),len(b))==counts
            assert len(np.unique(np.r_[a,b]))==sum(counts)
    assert file_hash(ROOT/'data/cifar-100-python/train')==assets['cifar_train']
    assert file_hash(ROOT/'data/sst2/train.parquet')==assets['sst2_train']
    from exp14c.cv_model import checkpoint_path
    assert file_hash(checkpoint_path())==assets['cv_pretrained']
    for name,h in assets['nlp_pretrained'].items(): assert file_hash(ROOT/'cache/bert-tiny'/name)==h
    checks=[]
    for task in ('cv','nlp'):
        for method in METHODS:
            base=next(c for c in grid(task) if c.method==method)
            for stage in ('search','final'):
                cfg=replace(base,stage=stage,seed=20261111 if stage=='final' else base.seed)
                directory=reference_dir(cfg)
                d,S,W=build_matrices(cfg.noise,cfg.total_steps,4,.9)
                with np.load(directory/'matrices.npz') as saved:
                    for k,x in [('coefficients',d),('strategy',S),('workload',W)]: np.testing.assert_array_equal(x,saved[k])
                for C in sorted({c.C for c in grid(task) if c.method==method}):
                    trial=replace(cfg,C=C);cal=calibrate(S,trial.protocol())
                    assert abs(epsilon_from_mu(C*cal['sensitivity']/cal['innovation_std_sum'],1e-5)-8)<1e-7
                    assert np.isclose(cal['innovation_std_sum']/C,calibrate(S,replace(cfg,C=1).protocol())['innovation_std_sum'])
                params=[torch.zeros(3,4),torch.zeros(5)];sigma=cal['innovation_std_sum']
                a=BandInvMFNoise(params,d,sigma,cfg.total_steps,cfg.seed+1)
                b=original['BandInvMFNoise'](params,d,sigma,cfg.total_steps,cfg.seed+1)
                for _ in range(cfg.total_steps):
                    for x,y in zip(a.next(),b.next()): torch.testing.assert_close(x,y,rtol=0,atol=0)
                checks.append(dict(task=task,method=method,stage=stage,steps=cfg.total_steps,matrix_exact=True,noise_exact=True,all_C_epsilon8=True))
                print(task,method,stage,'PASS',flush=True)
            for seed in FINAL_SEEDS:
                c=replace(base,stage='final',seed=seed);refdir=reference_dir(c)
                with np.load(historical/f'{task}_split.npz') as split: indices=np.arange(50000) if task=='cv' else split['train'].copy()
                order=indices[torch.randperm(len(indices),generator=torch.Generator().manual_seed(seed)).numpy()]
                np.testing.assert_array_equal(order,np.load(refdir/'train_order.npy'))
    from exp14c.pipeline import rank
    fake=[dict(accuracy=.5,epochs=[dict(validation_loss=.2)],trial_id='b'),dict(accuracy=.5,epochs=[dict(validation_loss=.2)],trial_id='a'),dict(accuracy=.5,epochs=[dict(validation_loss=.3)],trial_id='c')]
    assert [r['trial_id'] for r in rank(fake)]==['a','b','c']
    from exp14c.prepare import prepare
    prepare()
    save_json(RESULTS/'search_grid.json',[c.protocol()|dict(trial_id=c.id) for t in ('cv','nlp') for c in grid(t)])
    save_json(RESULTS/'source_provenance.json',code_hashes())
    assert protected_snapshot()==json.loads((RESULTS/'protected_before.json').read_text())
    save_json(RESULTS/'tests.json',dict(status='passed',optimizer_reference=True,checkpoint_resume=True,splits=True,all_80_formal_orders_exact=True,raw_assets=True,ranking=True,checks=checks))
if __name__=='__main__': main()
