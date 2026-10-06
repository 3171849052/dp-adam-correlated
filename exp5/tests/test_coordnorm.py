from exp5 import runtime
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from exp5.config import FIXED, METHODS, SEARCH_SEED, FINAL_SEEDS, Trial
from exp5.runtime import EXP, output_path
from exp5.mechanism import coordinate_transform, transform_clip, PerExample, clip_microbatch, LogicalSGDM
from exp5.privacy import build, materialize
from exp5.search import StagedSearch, winner, local_points
from exp5.final_runner import report
from exp5.train import DIAGNOSTICS, epoch_diagnostics
from exp5.probe import histogram_quantiles


def test_coordinate_transform_exact_and_per_coordinate_per_sample():
    g = torch.tensor([[0., -2., 7.], [4., .01, -8.]], dtype=torch.float64)
    tau = .3
    h = coordinate_transform(g, tau)
    assert torch.equal(h, g / (g.abs() + tau))
    assert not torch.allclose(h.mean(0), coordinate_transform(g.mean(0), tau))
    assert not torch.allclose(h, g / (g.norm(dim=1, keepdim=True) + tau))
    assert not torch.equal(h, g.sign())


def test_global_clip_across_complete_transformed_vector():
    g = dict(a=torch.tensor([[3., 0.], [0., 0.]]), b=torch.tensor([[4.], [0.]]))
    tau, C = 1., .5
    raw, h, factors = transform_clip(g, tau, C)
    expected_h = torch.tensor([.75, 0., .8])
    torch.testing.assert_close(raw, torch.tensor([5., 0.]))
    torch.testing.assert_close(h, torch.tensor([expected_h.norm(), 0.]))
    assert factors[1] == 1
    vector = torch.cat((g['a'][0], g['b'][0]))
    torch.testing.assert_close(vector, expected_h * C / expected_h.norm())
    assert float(vector.norm()) == pytest.approx(C)


def explicit(model, x, y, tau, C):
    sums, raw_sums = [torch.zeros_like(p) for p in model.parameters()], [torch.zeros_like(p) for p in model.parameters()]
    raw_norms, h_norms, factors = [], [], []
    for xx, yy in zip(x, y):
        gradients = torch.autograd.grad(nn.functional.cross_entropy(model(xx[None]), yy[None]), tuple(model.parameters()))
        transformed = [g/(g.abs()+tau) for g in gradients]
        norm = torch.cat([g.flatten() for g in transformed]).norm()
        factor = min(1., C/float(norm))
        raw_norms.append(float(torch.cat([g.flatten() for g in gradients]).norm()))
        h_norms.append(float(norm)); factors.append(factor)
        for s, r, g, h in zip(sums, raw_sums, gradients, transformed):
            s.add_(h*factor); r.add_(g)
    return sums, raw_sums, np.mean(raw_norms), np.mean(h_norms), np.mean(factors)


@pytest.mark.parametrize('chunk', [1, 7, 50, 100])
def test_chunked_matches_explicit_per_example(chunk):
    torch.manual_seed(8)
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3)).double()
    x, y = torch.randn(100, 4, dtype=torch.float64), torch.randint(3, (100,))
    sums, raws, raw_norm, h_norm, factor = explicit(model, x, y, .2, 1.)
    stats, raw_sums = clip_microbatch(PerExample(model, chunk), x, y, .2)
    for p, s, r, rr in zip(model.parameters(), sums, raw_sums, raws):
        torch.testing.assert_close(p.grad, s)
        torch.testing.assert_close(r, rr)
    assert stats['raw_norm_sum']/100 == pytest.approx(raw_norm)
    assert stats['h_norm_sum']/100 == pytest.approx(h_norm)
    assert stats['factor_sum']/100 == pytest.approx(factor)
    assert stats['correct'] == int((model(x).argmax(1) == y).sum())


class CountingNoise:
    def __init__(self): self.calls=0
    def next(self):
        self.calls += 1
        return [torch.tensor([7., -4.])]
    def marginal_std(self, step): return 1.


def test_1000_is_ten_physical_batches_noise_once_and_noisy_momentum_only():
    p = nn.Parameter(torch.tensor([1., 2.]))
    position = p.detach().clone()
    noise = CountingNoise()
    logical = LogicalSGDM([p], noise, .1)
    assert logical.accumulation == 10 and FIXED['accumulation'] == 10
    for step in range(2):
        old = logical.momentum[0].clone()
        for i in range(10):
            p.grad = torch.tensor([100., 200.])
            metrics = logical.finish_microbatch([p.grad*2], 100*np.sqrt(5))
            if i < 9:
                assert metrics is None and noise.calls == step
                torch.testing.assert_close(logical.momentum[0], old)
        dp = torch.tensor([1., 2.]) + torch.tensor([7., -4.])
        torch.testing.assert_close(logical.momentum[0], .9*old+dp)
        position -= .1*logical.momentum[0]
        torch.testing.assert_close(p, position)
        assert noise.calls == step+1
        assert metrics['raw_mean_gradient_norm'] == pytest.approx(2*np.sqrt(5))
        assert metrics['batch_coherence'] == pytest.approx(1.)
    with pytest.raises(ValueError, match='physical batch'):
        logical.finish_microbatch([p.grad], 1., 99)


def test_ten_physical_chunks_match_full_logical_explicit():
    torch.manual_seed(2)
    model = nn.Linear(2, 2).double()
    x, y = torch.randn(1000, 2, dtype=torch.float64), torch.randint(2, (1000,))
    expected, _, _, _, _ = explicit(model, x, y, .7, 1.)
    sums = [torch.zeros_like(p) for p in model.parameters()]
    for xx, yy in zip(x.chunk(10), y.chunk(10)):
        stats, _ = clip_microbatch(PerExample(model, 13), xx, yy, .7)
        assert stats['count'] == 100
        for s, p in zip(sums, model.parameters()): s.add_(p.grad)
    for s, e in zip(sums, expected): torch.testing.assert_close(s/1000, e/1000)


def test_iid_replace_one_sparse_accounting():
    _, strategy, _, meta = build(METHODS[0], 1.)
    np.testing.assert_array_equal(strategy, np.eye(250))
    assert meta['per_step_sensitivity'] == 2/1000
    assert meta['sensitivity'] == pytest.approx(.002*np.sqrt(5))
    assert meta['innovation_std'] == pytest.approx(.002*np.sqrt(5)/meta['mu'])
    assert meta['direct_participations'] == 5 and not meta['sampling_amplification']


def test_bandinvmf_exp2_momentum_bandwidth_four_and_sparse_accountant():
    from exp2.bandinvmf import build_matrices
    from exp2.privacy import fixed_epoch_sensitivity
    coef, strategy, workload, meta = build(METHODS[1], 1.)
    c, s, w = build_matrices('momentum_bandinvmf', 250, 4, .9)
    np.testing.assert_array_equal(coef, c)
    np.testing.assert_array_equal(strategy, s)
    np.testing.assert_allclose(workload, w)
    assert workload[1] == pytest.approx(1.9)
    assert meta['num_bands'] == 4 and meta['workload'] == 'momentum'
    assert meta['strategy_sensitivity'] == fixed_epoch_sensitivity(strategy, 5, 50)
    np.testing.assert_allclose(materialize(coef, 250) @ strategy, np.eye(250), atol=1e-12)


def test_local_assets_no_network(monkeypatch):
    import socket
    from torchvision.datasets import CIFAR100
    from exp5.model import pretrained_vit
    from exp5.runtime import ROOT
    def deny(*args, **kwargs): raise AssertionError('Network is forbidden')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    model, metadata = pretrained_vit()
    assert all(p.requires_grad for p in model.parameters())
    assert (ROOT/metadata['checkpoint_path']).resolve().is_relative_to(ROOT/'cache')
    assert len(CIFAR100(ROOT/'data', train=True, download=False)) == 50000
    assert len(CIFAR100(ROOT/'data', train=False, download=False)) == 10000


def test_output_paths_identity_and_C_fixed(tmp_path):
    assert tmp_path.resolve().is_relative_to(EXP)
    assert output_path(tmp_path) == tmp_path
    with pytest.raises(ValueError, match='Output must be inside'): output_path(EXP.parent/'other')
    base = Trial(METHODS[0], SEARCH_SEED, .001, .01)
    assert base.identity == Trial(METHODS[0], SEARCH_SEED, 1e-3, 1e-2).identity
    candidates = [base, Trial(METHODS[1], SEARCH_SEED, .001, .01),
                  Trial(METHODS[0], SEARCH_SEED+1, .001, .01),
                  Trial(METHODS[0], SEARCH_SEED, .002, .01), Trial(METHODS[0], SEARCH_SEED, .001, .02)]
    assert len({t.identity for t in candidates}) == 5
    with pytest.raises(ValueError, match='C=1'): Trial(METHODS[0], SEARCH_SEED, .001, .01, 2.)


def synthetic_summary(t, top1=.2):
    return dict(status='completed', smoke=False, trial_id=t.identity, fixed=FIXED,
                **t.asdict(), optimizer_steps=250, noise_steps=250, physical_batches=2500,
                epochs=[dict(logical_steps=s) for s in (50,100,150,200,250)],
                final=dict(epsilon=8., test_top1=top1, train_top1=.1, train_loss=2., test_loss=3.),
                diagnostics=dict.fromkeys(DIAGNOSTICS, .1), privacy=build(t.method, t.C)[3])


def test_search_stages_resume_dedup_freeze_and_ties(tmp_path):
    stages, executed = [], []
    grid=[.01,.02,.04,.08,.16]
    def fake_launch(jobs, on_complete):
        stages.append([j['trial'] for j in jobs])
        for j in jobs:
            t=Trial(**j['trial']); folder=Path(j['result_dir'])
            if not (folder/'summary.json').exists():
                folder.mkdir(parents=True)
                score=.5-.01*abs(np.log2(t.tau/.04))-.01*abs(np.log2(t.lr/.002))
                (folder/'summary.json').write_text(json.dumps(synthetic_summary(t,score)))
                executed.append(t.identity)
            on_complete(j)
    selected=StagedSearch(tmp_path,fake_launch).run(dict(tau_grid=grid))
    assert len(stages)==5
    assert [t['tau'] for t in stages[0]] == grid and all(t['lr']==.001 for t in stages[0])
    assert [t['lr'] for t in stages[1]] == [.00025,.0005,.001,.002,.004,.008]
    assert {t['tau'] for t in stages[2]} == {.02,.04,.08} and len(stages[2])==9
    assert [t['lr'] for t in stages[3]] == [.0005,.001,.002,.004,.008]
    assert all(t['tau']==.04 for stage in stages[3:] for t in stage)
    assert all(t['C']==1. for stage in stages for t in stage)
    n=len(executed)
    assert n==len(set(executed))==selected['total_completed_search_trials']
    StagedSearch(tmp_path,fake_launch).run(dict(tau_grid=grid))
    assert len(executed)==n
    assert len(list(tmp_path.glob('stage*_summary.json')))==5
    assert (tmp_path/'iid_frozen.json').exists() and (tmp_path/'bandinvmf_frozen.json').exists()
    ties=[dict(tau=tau,lr=lr,final_test_top1=.2) for tau,lr in [(1.,.02),(1.,.01),(2.,.01)]]
    assert winner(ties)==ties[2]
    assert len(local_points(dict(tau=.01,lr=.001),grid))==6


def test_final_report_ddof_one(tmp_path):
    selected=dict(methods={m:dict(C=1.,lr=.001,tau=.04) for m in METHODS},total_completed_search_trials=24)
    rows=[synthetic_summary(Trial(m,seed,.001,.04),score) for m in METHODS for seed,score in zip(FINAL_SEEDS,(.1,.2,.3))]
    result=report(rows,selected,tmp_path)
    for r in result['methods'].values():
        assert r['mean_top1']==pytest.approx(.2) and r['sample_std_top1']==pytest.approx(.1)
        assert r['selected_tau']==.04 and len(r['seeds'])==3
        assert 'batch_coherence' in r['mean_diagnostics']


def test_probe_approximation_and_no_full_gradient_storage():
    q=histogram_quantiles([1,1,1,1],lower=-4,upper=0)
    assert q==sorted(q) and 1e-4<q[0]<q[-1]<1


def test_launcher_three_gpus_single_trial_each(monkeypatch,tmp_path):
    from exp5 import launch_batch
    live, started, peaks = {}, [], []
    class Process:
        def __init__(self,command,cwd,env,stdout,stderr):
            self.gpu=env['CUDA_VISIBLE_DEVICES']; assert self.gpu not in live
            assert '--tau' in command and command[command.index('--C')+1]=='1.0'
            self.pid=len(started)+1; self.polls=0
            live[self.gpu]=self; started.append(self.gpu); peaks.append(len(live))
            folder=Path(command[command.index('--result-dir')+1])
            (folder/'summary.json').write_text(json.dumps(dict(status='completed')))
        def poll(self):
            self.polls+=1
            if self.polls<2: return None
            del live[self.gpu]
            return 0
        def terminate(self): del live[self.gpu]
        def wait(self): return 0
    monkeypatch.setattr(launch_batch.fcntl,'flock',lambda *args:None)
    monkeypatch.setattr(launch_batch.subprocess,'Popen',Process)
    monkeypatch.setattr(launch_batch.time,'sleep',lambda _:None)
    jobs=[dict(trial=Trial(METHODS[0],SEARCH_SEED+i,.001,.04).asdict(),result_dir=str(tmp_path/f'trial_{i}'),smoke=True) for i in range(6)]
    launch_batch.launch(jobs)
    assert started==['0','1','2','0','1','2'] and max(peaks)==3 and not live


def test_epoch_norm_statistics_include_between_step_variation():
    steps=[dict.fromkeys(DIAGNOSTICS,0.) for _ in range(2)]
    # Four transformed norms [1,1,3,3]: within-step std=0, epoch std=1.
    totals=dict(count=4,raw_norm_sum=8.,h_norm_sum=8.,h_norm_sq_sum=20.,clipped=2,factor_sum=3.)
    metrics=epoch_diagnostics(totals,steps)
    record=dict(epoch=1,**metrics)
    assert record['mean_transformed_sample_norm']==2.
    assert record['transformed_norm_std']==1.
    assert record['clip_fraction']==.5 and record['mean_clip_factor']==.75
