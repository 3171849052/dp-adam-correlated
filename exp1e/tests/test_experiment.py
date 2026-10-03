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
from exp1e.model import ViTTiny, pretrained_vit, initialization_digest
from exp1e.train import ROOT, load_config, seed_all, image_transforms
from exp1e.bandinvmf import build_matrices, BandInvMFNoise
from exp1e.privacy import calibrate, gdp_delta
from exp1e.scale import LogicalBatch, ScaledGhostModule, freeze_adam_scale, clipped_microbatch
from exp1e.diagnostics import EpochDiagnostics, statistics
from exp1e.sweep import stage1_trials, stage2_trials, final_trials, selected_configs, FINAL_SEEDS, LR_GRID, SCALE

torch.set_num_threads(2)
CONFIG = load_config(ROOT / 'exp1e/config.yaml')


def test_config_grid_calibration():
    assert (CONFIG['privacy']['k'], CONFIG['privacy']['b_participation'], CONFIG['total_steps']) == (5,50,250)
    assert (CONFIG['physical_batch_size'], CONFIG['logical_batch_size'], CONFIG['gradient_accumulation']) == (250,1000,4)
    jobs = stage1_trials()
    assert len(jobs)==5 and [j['max_grad_norm'] for j in jobs]==[100,150,200,300,500]
    assert all(j['lr']==1e-3 and j['eps_scale']==.1 and j['num_bands']==4 and j['seed']==20261001 for j in jobs)
    assert tuple(j['lr'] for j in stage2_trials(150))==LR_GRID==(5e-4,1e-3,2e-3,3e-3)
    finals = final_trials(selected_configs(150,2e-3))
    assert len(finals)==12 and Counter(j['seed'] for j in finals)=={s:4 for s in FINAL_SEEDS}
    assert FINAL_SEEDS==(20261011,20261012,20261013)
    d,c,w=build_matrices(SCALE,250,4,.9)
    from exp1d.bandinvmf import build_matrices as original
    for a,b in zip((d,c,w),original(SCALE,250,4,.9)): np.testing.assert_array_equal(a,b)
    stds=[]
    for job in jobs:
        config=copy.deepcopy(CONFIG); config['privacy']['max_grad_norm']=job['max_grad_norm']
        result=calibrate(c,config)
        assert gdp_delta(result['target_mu'],8)==pytest.approx(1e-5)
        assert result['innovation_std_sum']==pytest.approx(job['max_grad_norm']*result['sensitivity']/result['target_mu'])
        stds.append(result['innovation_std_sum'])
    assert len(set(stds))==5
    np.testing.assert_allclose(np.array(stds)/stds[0], [1,1.5,2,3,5])
    d,c,_=build_matrices('dp_adam',250,4,.9)
    np.testing.assert_array_equal(d,[1.])
    np.testing.assert_array_equal(c,np.eye(250))


@pytest.mark.parametrize("seed", [20261001, *FINAL_SEEDS])
def test_initialization_full_finetuning_rng(seed):
    from exp1c.model import pretrained_vit as original
    from exp1c.train import image_transforms as original_transforms
    os.environ['HF_HOME']=str(ROOT/'exp1e/cache/huggingface')
    os.environ['TORCH_HOME']=str(ROOT/'exp1e/cache/torch')
    seed_all(seed)
    reference,cfg=original()
    digest=initialization_digest(reference)
    rng=torch.get_rng_state().clone()
    from PIL import Image
    image=Image.fromarray(np.random.default_rng(0).integers(0,256,(32,32,3),dtype=np.uint8))
    expected_augmentation=image_transforms(cfg)[0](image)
    for job in final_trials(selected_configs(100,1e-3)):
        seed_all(seed)
        model,new_cfg=pretrained_vit()
        assert initialization_digest(model)==digest
        assert all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters())
        assert model.head.out_features==100 and cfg==new_cfg
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        torch.testing.assert_close(image_transforms(new_cfg)[0](image),expected_augmentation,rtol=0,atol=0)
        order=torch.randperm(CONFIG['dataset_size'],generator=torch.Generator().manual_seed(seed))
        if job==final_trials(selected_configs(100,1e-3))[0]: expected_order=order
        torch.testing.assert_close(order,expected_order,rtol=0,atol=0)
    assert [repr(t) for t in image_transforms(cfg)]==[repr(t) for t in original_transforms(cfg)]


def tiny():
    return ViTTiny(patch_size=4,image_size=16,embed_dim=12,depth=1,heads=3,mlp_ratio=2,num_classes=3)


@pytest.mark.parametrize('scaled',[False,True])
def test_four_microbatches_one_noise_and_previous_state(scaled):
    model=tiny()
    opt=torch.optim.Adam(model.parameters(),betas=(.9,.999))
    d,_,_=build_matrices('dp_adam_bandinvmf_scale',2,4,.9)
    noise=BandInvMFNoise(model.parameters(),d,1.,2,10)
    logical=LogicalBatch(model,opt,4,1000,noise,scaled=scaled,eps_scale=.01)
    for step in range(2):
        expected=[torch.zeros_like(p) if step==0 else opt.state[p]['exp_avg_sq']/(1-.999**step) for p in model.parameters()]
        for i in range(4):
            logical.begin_microbatch()
            if scaled:
                for p,vhat in zip(model.parameters(),expected):
                    torch.testing.assert_close(p._logical_sqrt_vhat,vhat.sqrt())
                    torch.testing.assert_close(p._logical_scale,1/(vhat.sqrt()+.01))
            for p in model.parameters(): p.grad=torch.ones_like(p)
            assert logical.finish_microbatch()==(i==3)
            assert noise.step_count==logical.optimizer_steps==step+int(i==3)
    assert all(int(opt.state[p]['step'])==2 for p in model.parameters())


@pytest.mark.parametrize('clip',[100.,150.,200.,300.,500.])
def test_scale_norms_clipping_and_statistics(clip):
    torch.manual_seed(42)
    base=tiny()
    wrapped=ScaledGhostModule(copy.deepcopy(base),clip)
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
        for total,p in zip(sums,base.parameters()): total.add_(p.grad,alpha=min(1.,clip/float(s)))
    stats=EpochDiagnostics(clip)
    stats.add_state(list(wrapped.parameters()))
    _,clipped=clipped_microbatch(wrapped,x,y,clip,stats)
    norms,coords=stats.aggregate()
    assert clipped==sum(s>clip for s in scaled)
    np.testing.assert_allclose(stats.norms['unscaled_norm'][0],unscaled,rtol=5e-5)
    np.testing.assert_allclose(stats.norms['scaled_norm'][0],scaled,rtol=5e-5)
    for p,total in zip(wrapped.parameters(),sums): torch.testing.assert_close(p.grad,total,rtol=5e-5,atol=1e-6)
    assert set(wrapped.trainable_parameters)==set(wrapped.parameters())
    for group,values in [('unscaled_norm',unscaled),('scaled_norm',scaled)]:
        for q in (10,25,50,75,90,99): assert norms[group][f'p{q}']==pytest.approx(np.quantile(values,q/100),rel=5e-5)
    for group,attr in [('sqrt_vhat','_logical_sqrt_vhat'),('scale','_logical_scale')]:
        values=np.concatenate([getattr(p,attr).numpy().ravel() for p in wrapped.parameters()])
        assert coords[group]==statistics([values])


def test_no_hardcoded_participation_and_fixed_bands(tmp_path):
    import yaml
    raw=yaml.safe_load((ROOT/'exp1e/config.yaml').read_text())
    assert 'gradient_accumulation' not in raw
    assert 'total_steps' not in raw and 'k' not in raw['privacy'] and 'b_participation' not in raw['privacy']
    raw['bandinvmf']['num_bands']=3
    path=tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(raw))
    with pytest.raises(AssertionError): load_config(path)
