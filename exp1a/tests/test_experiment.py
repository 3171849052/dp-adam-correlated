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
from torch import nn
from opacus.grad_sample import GradSampleModuleFastGradientClipping
import yaml
from exp1a.model import ViTTiny, pretrained_vit, initialization_digest
from exp1a.clipping import LogicalBatch, IIDNoise, EpochNorms, clipped_microbatch
from exp1a.privacy import calibrate, gdp_delta, epsilon_from_mu
from exp1a.train import ROOT, load_config, seed_all, image_transforms
from exp1a.sweep import trials

torch.set_num_threads(2)
CONFIG = load_config(ROOT / 'exp1a/config.yaml')


def test_derived_config():
    raw = yaml.safe_load((ROOT / 'exp1a/config.yaml').read_text())
    assert 'total_steps' not in raw and 'k' not in raw['privacy'] and 'b_participation' not in raw['privacy']
    assert (CONFIG['privacy']['k'], CONFIG['privacy']['b_participation'], CONFIG['total_steps']) == (5, 50, 250)
    assert (CONFIG['physical_batch_size'], CONFIG['gradient_accumulation']) == (50, 20)


@pytest.mark.parametrize('clip', [.3, 1., 3., 10.])
def test_calibration_override(clip):
    cfg = copy.deepcopy(CONFIG)
    cfg['privacy']['max_grad_norm'] = clip
    result = calibrate(np.eye(cfg['total_steps']), cfg)
    assert result['sensitivity'] == pytest.approx(np.sqrt(5))
    assert result['innovation_std_sum'] == pytest.approx(clip*np.sqrt(5)/result['target_mu'])
    assert gdp_delta(result['target_mu'], 8) == pytest.approx(1e-5)
    assert epsilon_from_mu(result['target_mu'], 1e-5) == pytest.approx(8)
    assert not result['sampling_amplification']


def tiny():
    return ViTTiny(patch_size=4, image_size=16, embed_dim=12, depth=1, heads=3, mlp_ratio=2, num_classes=3)


def explicit(model, x, y):
    norms = dict(full_model_norm=[], backbone_norm=[], head_norm=[])
    sums = [torch.zeros_like(p) for p in model.parameters()]
    for xi, yi in zip(x, y):
        model.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(xi[None]), yi[None]).backward()
        head = sum(p.grad.square().sum() for name, p in model.named_parameters() if name.startswith('head.'))
        backbone = sum(p.grad.square().sum() for name, p in model.named_parameters() if not name.startswith('head.'))
        full = (head+backbone).sqrt()
        for key, value in [('full_model_norm',full), ('head_norm',head.sqrt()), ('backbone_norm',backbone.sqrt())]:
            norms[key].append(value.item())
        coef = min(1., .3/full.item())
        for total, p in zip(sums, model.parameters()):
            total.add_(p.grad, alpha=coef)
    return {key: np.array(values) for key, values in norms.items()}, sums


def test_exact_preclipping_norms_and_quantiles():
    torch.manual_seed(42)
    base = tiny()
    model = GradSampleModuleFastGradientClipping(copy.deepcopy(base), loss_reduction='sum', max_grad_norm=.3)
    x, y = torch.randn(3,3,16,16), torch.tensor([0,1,2])
    expected, sums = explicit(base, x, y)
    _, norms = clipped_microbatch(model, x, y, .3)
    for key in norms:
        np.testing.assert_allclose(norms[key], expected[key], rtol=5e-5, atol=1e-6)
    np.testing.assert_allclose(norms['full_model_norm']**2, norms['head_norm']**2+norms['backbone_norm']**2, rtol=1e-6)
    for p, total in zip(model.parameters(), sums):
        torch.testing.assert_close(p.grad, total, rtol=5e-5, atol=1e-6)
    stats = EpochNorms(.3); stats.add(norms)
    aggregated = stats.aggregate()
    for key in norms:
        for q in [10,25,50,75,90,99]:
            assert aggregated[key][f'p{q}'] == pytest.approx(np.quantile(expected[key],q/100), rel=5e-5)
        assert aggregated[key]['clip_fraction'] == float((expected[key]>.3).mean())
    assert set(model.trainable_parameters) == set(model.parameters())
    assert all(p.grad is not None and p._norm_sample.shape == (3,) and p.requires_grad for p in model.parameters())


def test_twenty_microbatches_one_noise_and_step():
    model = tiny()
    optimizer = torch.optim.Adam(model.parameters())
    noise = IIDNoise(model.parameters(), 2., 99)
    logical = LogicalBatch(model, optimizer, 20, 1000, noise)
    for i in range(20):
        logical.begin_microbatch()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        assert logical.finish_microbatch() == (i==19)
        assert noise.step_count == logical.optimizer_steps == int(i==19)
    assert all(int(optimizer.state[p]['step']) == 1 for p in model.parameters())


def test_zero_noise_dp_equals_clipped_path():
    torch.manual_seed(10)
    base = tiny()
    models = [GradSampleModuleFastGradientClipping(copy.deepcopy(base), loss_reduction='sum', max_grad_norm=1.) for _ in range(2)]
    assert initialization_digest(models[0]) == initialization_digest(models[1])
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in models]
    noise = IIDNoise(models[0].parameters(), 0., 17)
    batches = [LogicalBatch(m, opt, 20, 40, noise if i==0 else None) for i,(m,opt) in enumerate(zip(models,optimizers))]
    for step in range(2):
        x,y = torch.randn(40,3,16,16), torch.randint(3,(40,))
        for i in range(20):
            for m,logical in zip(models,batches):
                logical.begin_microbatch()
                clipped_microbatch(m,x[i*2:i*2+2],y[i*2:i*2+2],1.)
                logical.finish_microbatch()
        for a,b in zip(models[0].parameters(),models[1].parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert noise.step_count == 2


def test_initialization_matches_exp1_and_all_methods():
    # Import the reference without directing downloads or new files into exp1.
    from exp1.model import pretrained_vit as original
    os.environ['HF_HOME'] = str(ROOT / 'exp1a/cache/huggingface')
    os.environ['TORCH_HOME'] = str(ROOT / 'exp1a/cache/torch')
    seed_all(CONFIG['seed'])
    reference,cfg = original()
    expected = initialization_digest(reference)
    for method in ('adam','dp_adam','clipped_adam_no_noise'):
        seed_all(CONFIG['seed'])
        model,new_cfg = pretrained_vit()
        assert initialization_digest(model) == expected
        assert cfg == new_cfg
        assert all(p.requires_grad for p in model.parameters())
        assert model.head.out_features == 100
        if method != 'adam':
            wrapped = GradSampleModuleFastGradientClipping(model,loss_reduction='sum')
            x,y = torch.randn(1,3,224,224),torch.tensor([2])
            clipped_microbatch(wrapped,x,y,1.)
            assert set(wrapped.trainable_parameters) == set(model.parameters())
            assert all(p.grad is not None and p._norm_sample.shape==(1,) for p in model.parameters())
            assert model.cls.weight._norm_sample.item()>0
            assert model.position.weight._norm_sample.item()>0


def test_transforms_equal_exp1():
    from exp1.train import image_transforms as original
    cfg = dict(mean=(.5,)*3, std=(.5,)*3, crop_pct=.9)
    assert [repr(t) for t in image_transforms(cfg)] == [repr(t) for t in original(cfg)]


def test_grid():
    jobs=trials()
    assert len(jobs)==27
    assert Counter(j['method'] for j in jobs)==dict(adam=3,dp_adam=12,clipped_adam_no_noise=12)
    assert len({(j['method'],j['name']) for j in jobs})==27
    assert {j['lr'] for j in jobs}=={1e-4,3e-4,1e-3}
    assert {j['max_grad_norm'] for j in jobs if j['method']!='adam'}=={.3,1.,3.,10.}


@pytest.mark.parametrize('exit_code',[0,1])
def test_launcher_four_gpu_queue(tmp_path,monkeypatch,exit_code):
    import exp1a.sweep as sweep
    launched,live=[],set()
    class Process:
        def __init__(self,cmd,cwd,env,stdout,stderr):
            self.gpu=env['CUDA_VISIBLE_DEVICES']
            assert self.gpu not in live
            live.add(self.gpu)
            assert len(live)<=4
            launched.append(self.gpu)
            directory=Path(cmd[cmd.index('--result-dir')+1])
            if exit_code==0:
                (directory/'summary.json').write_text(json.dumps(dict(calibration=None,final_norm_stats=None,
                    epochs=[dict(epoch=1,test_top1=.1)],final=dict(test_top1=.1,train_loss=4.,clip_fraction=None,noise_std=0.),
                    wall_seconds=1.,initialization_sha256='same')))
            self.polls=0
        def poll(self):
            self.polls+=1
            if self.polls==1: return None
            live.remove(self.gpu)
            return exit_code
    monkeypatch.setattr(sweep,'ROOT',tmp_path)
    monkeypatch.setattr(sweep.subprocess,'Popen',Process)
    monkeypatch.setattr(sweep.time,'sleep',lambda _:None)
    assert sweep.main(Namespace(config=ROOT/'exp1a/config.yaml'))==exit_code
    assert len(launched)==27 and not live and set(launched)=={'0','1','2','3'}
    rows=list(csv.DictReader((tmp_path/'exp1a/results/sweep_summary.csv').open()))
    assert len(rows)==27
    assert all(row['status']==('failed' if exit_code else 'completed') for row in rows)


def test_pretrained_exact_norms_against_individual_gradients():
    seed_all(CONFIG['seed'])
    base, _ = pretrained_vit()
    model = GradSampleModuleFastGradientClipping(copy.deepcopy(base), loss_reduction='sum', max_grad_norm=.3)
    x, y = torch.randn(2,3,224,224), torch.tensor([0,3])
    expected, sums = explicit(base,x,y)
    _, norms = clipped_microbatch(model,x,y,.3)
    for key in norms:
        np.testing.assert_allclose(norms[key],expected[key],rtol=5e-5,atol=1e-5)
    for p, total in zip(model.parameters(),sums):
        torch.testing.assert_close(p.grad,total,rtol=5e-5,atol=1e-6)
