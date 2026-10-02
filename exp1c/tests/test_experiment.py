import copy
import csv
import json
import os
from argparse import Namespace
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
import torch
from exp1c.model import ViTTiny, pretrained_vit, initialization_digest
from exp1c.train import ROOT, load_config, seed_all, image_transforms
from exp1c.bandinvmf import build_matrices, BandInvMFNoise
from exp1c.privacy import calibrate, gdp_delta
from exp1c.scale import LogicalBatch, ScaledGhostModule, freeze_adam_scale, clipped_microbatch
from exp1c.diagnostics import EpochDiagnostics, statistics
from exp1c.sweep import trials

torch.set_num_threads(2)
CONFIG = load_config(ROOT / 'exp1c/config.yaml')


def test_config_grid_calibration():
    assert (CONFIG['privacy']['k'], CONFIG['privacy']['b_participation'], CONFIG['total_steps']) == (5,50,250)
    assert (CONFIG['physical_batch_size'], CONFIG['gradient_accumulation']) == (50,20)
    jobs = trials()
    assert len(jobs) == 12
    assert Counter(j['method'] for j in jobs) == {'dp_adam_bandinvmf_momentum':3, 'dp_adam_bandinvmf_scale':9}
    assert len({(j['method'],j['name']) for j in jobs}) == 12
    assert {j['num_bands'] for j in jobs} == {4}
    assert {j['max_grad_norm'] for j in jobs} == {1.}
    assert {j['lr'] for j in jobs} == {5e-4,1e-3,3e-3}
    assert {(j['lr'], j['eps_scale']) for j in jobs if j['eps_scale'] is not None} == {(lr,eps) for lr in (5e-4,1e-3,3e-3) for eps in (1e-3,1e-2,1e-1)}
    for method in {j['method'] for j in jobs}:
        from exp1.bandinvmf import build_matrices as original
        d,c,w=build_matrices(method,CONFIG['total_steps'],4,.9)
        for a,b in zip((d,c,w),original(method,CONFIG['total_steps'],4,.9)):
            np.testing.assert_array_equal(a,b)
        result=calibrate(c,CONFIG)
        assert gdp_delta(result['target_mu'],8)==pytest.approx(1e-5)
        assert result['innovation_std_sum']==pytest.approx(result['sensitivity']/result['target_mu'])
        assert not result['sampling_amplification']


def test_initialization_full_finetuning_rng():
    from exp1.model import pretrained_vit as original
    from exp1.train import image_transforms as original_transforms
    os.environ['HF_HOME']=str(ROOT/'exp1c/cache/huggingface')
    os.environ['TORCH_HOME']=str(ROOT/'exp1c/cache/torch')
    seed_all(CONFIG['seed'])
    reference,cfg=original()
    digest=initialization_digest(reference)
    rng=torch.get_rng_state().clone()
    for job in trials():
        seed_all(CONFIG['seed'])
        model,new_cfg=pretrained_vit()
        assert initialization_digest(model)==digest
        assert all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters())
        assert model.head.out_features==100 and cfg==new_cfg
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        order=torch.randperm(CONFIG['dataset_size'],generator=torch.Generator().manual_seed(CONFIG['seed']))
        if job==trials()[0]: expected_order=order
        torch.testing.assert_close(order,expected_order,rtol=0,atol=0)
    assert [repr(t) for t in image_transforms(cfg)]==[repr(t) for t in original_transforms(cfg)]


def tiny():
    return ViTTiny(patch_size=4,image_size=16,embed_dim=12,depth=1,heads=3,mlp_ratio=2,num_classes=3)


@pytest.mark.parametrize('scaled',[False,True])
def test_twenty_microbatches_one_noise_and_previous_state(scaled):
    model=tiny()
    opt=torch.optim.Adam(model.parameters(),betas=(.9,.999))
    d,_,_=build_matrices('dp_adam_bandinvmf_scale',2,4,.9)
    noise=BandInvMFNoise(model.parameters(),d,1.,2,10)
    logical=LogicalBatch(model,opt,20,1000,noise,scaled=scaled,eps_scale=.01)
    for step in range(2):
        expected=[torch.zeros_like(p) if step==0 else opt.state[p]['exp_avg_sq']/(1-.999**step) for p in model.parameters()]
        for i in range(20):
            logical.begin_microbatch()
            if scaled:
                for p,vhat in zip(model.parameters(),expected):
                    torch.testing.assert_close(p._logical_sqrt_vhat,vhat.sqrt())
                    torch.testing.assert_close(p._logical_scale,1/(vhat.sqrt()+.01))
            for p in model.parameters(): p.grad=torch.ones_like(p)
            assert logical.finish_microbatch()==(i==19)
            assert noise.step_count==logical.optimizer_steps==step+int(i==19)
    assert all(int(opt.state[p]['step'])==2 for p in model.parameters())


def test_scale_norms_clipping_and_statistics():
    torch.manual_seed(42)
    base=tiny()
    wrapped=ScaledGhostModule(copy.deepcopy(base),1.)
    opt=torch.optim.Adam(wrapped.parameters())
    # Coordinate-dependent completed vhat, rather than a constant initial scale.
    for p in wrapped.parameters():
        opt.state[p].update(step=torch.tensor(3.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.rand_like(p)*.001)
    freeze_adam_scale(opt,.01)
    x,y=torch.randn(3,3,16,16),torch.tensor([0,1,2])
    unscaled,scaled=[],[]
    sums=[torch.zeros_like(p) for p in base.parameters()]
    for xi,yi in zip(x,y):
        base.zero_grad()
        torch.nn.functional.cross_entropy(base(xi[None]),yi[None]).backward()
        u=sum(p.grad.square().sum() for p in base.parameters()).sqrt()
        s=sum((p.grad*q._logical_scale).square().sum() for p,q in zip(base.parameters(),wrapped.parameters())).sqrt()
        unscaled.append(float(u)); scaled.append(float(s))
        for total,p in zip(sums,base.parameters()): total.add_(p.grad,alpha=min(1.,1/float(s)))
    stats=EpochDiagnostics(1.)
    stats.add_state(list(wrapped.parameters()))
    _,clipped=clipped_microbatch(wrapped,x,y,1.,stats)
    norms,coords=stats.aggregate()
    assert clipped==sum(s>1 for s in scaled)
    np.testing.assert_allclose(stats.norms['unscaled_norm'][0],unscaled,rtol=5e-5)
    np.testing.assert_allclose(stats.norms['scaled_norm'][0],scaled,rtol=5e-5)
    for p,total in zip(wrapped.parameters(),sums): torch.testing.assert_close(p.grad,total,rtol=5e-5,atol=1e-6)
    assert set(wrapped.trainable_parameters)==set(wrapped.parameters())
    for group,values in [('unscaled_norm',unscaled),('scaled_norm',scaled)]:
        for q in (10,25,50,75,90,99): assert norms[group][f'p{q}']==pytest.approx(np.quantile(values,q/100),rel=5e-5)
    for group,attr in [('sqrt_vhat','_logical_sqrt_vhat'),('scale','_logical_scale')]:
        values=np.concatenate([getattr(p,attr).numpy().ravel() for p in wrapped.parameters()])
        assert coords[group]==statistics([values])


def fake_summary(job):
    return dict(status='completed',smoke=False,method=job['method'],lr=job['lr'],eps_scale=job['eps_scale'],
        wall_seconds=1.,initialization_sha256='same',calibration=dict(target_mu=1.,sensitivity=2.,innovation_std_sum=2.),
        epochs=[dict(epoch=1,test_top1=.1)],final=dict(test_top1=.1,train_loss=4.,clip_fraction=.5,marginal_noise_std=.01),
        final_norm_stats={g:{q:1. for q in ('p50','p90','p99')} for g in ('scaled_norm','unscaled_norm')},
        final_scale_stats={g:{q:1. for q in ('p50','p90')} for g in ('sqrt_vhat','scale')})


@pytest.mark.parametrize('code',[0,1])
def test_launcher_four_gpus_and_completed_skip(tmp_path,monkeypatch,code):
    import exp1c.sweep as sweep
    monkeypatch.setattr(sweep,'ROOT',tmp_path)
    monkeypatch.setattr(sweep.torch.cuda,'device_count',lambda:4)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(sweep.time,'sleep',lambda _:None)
    jobs=trials(); live=set(); launched=[]
    saved=tmp_path/'exp1c/results/sweep'/jobs[0]['method']/jobs[0]['name']
    saved.mkdir(parents=True)
    (saved/'summary.json').write_text(json.dumps(fake_summary(jobs[0])))
    initial=(saved/'summary.json').read_bytes()
    class Process:
        def __init__(self,cmd,cwd,env,stdout,stderr):
            self.gpu=env['CUDA_VISIBLE_DEVICES']; assert self.gpu not in live
            live.add(self.gpu); assert len(live)<=4
            launched.append(self.gpu); self.polls=0
            directory=Path(cmd[cmd.index('--result-dir')+1])
            assert stdout.name==str(directory/'train.log')
            if code==0:
                job=next(j for j in jobs if j['method']==directory.parent.name and j['name']==directory.name)
                (directory/'summary.json').write_text(json.dumps(fake_summary(job)))
        def poll(self):
            self.polls+=1
            if self.polls==1: return None
            live.remove(self.gpu); return code
    monkeypatch.setattr(sweep.subprocess,'Popen',Process)
    assert sweep.main(Namespace(config=ROOT/'exp1c/config.yaml'))==code
    assert len(launched)==11 and not live and set(launched)=={'0','1','2','3'}
    assert (saved/'summary.json').read_bytes()==initial
    rows=list(csv.DictReader((tmp_path/'exp1c/results/sweep_summary.csv').open()))
    assert len(rows)==12
    assert rows[0]['status']=='completed'
    assert all(row['status']==('failed' if code else 'completed') for row in rows[1:])


def test_no_hardcoded_participation_and_fixed_bands(tmp_path):
    import yaml
    raw=yaml.safe_load((ROOT/'exp1c/config.yaml').read_text())
    assert 'total_steps' not in raw and 'k' not in raw['privacy'] and 'b_participation' not in raw['privacy']
    raw['bandinvmf']['num_bands']=3
    path=tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(raw))
    with pytest.raises(AssertionError): load_config(path)
