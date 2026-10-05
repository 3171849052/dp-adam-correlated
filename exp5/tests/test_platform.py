import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from exp5.config import FIXED, METHODS, SEARCH_SEED, Trial
from exp5.runtime import EXP, output_path
from exp5.mechanism import LogicalSGDM, BandInvMFNoise, clip_microbatch
from exp5.privacy import build, materialize
from exp5.search import StagedSearch, winner
from exp5.final_runner import report


def wrapper(model, C):
    return GradSampleModuleFastGradientClipping(model, loss_reduction='sum',
                                              max_grad_norm=C, use_ghost_clipping=True)


def test_ten_microbatches_match_logical_and_explicit_global_clipping():
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3))
    x, y, C = torch.randn(1000, 4), torch.randint(3, (1000,)), .3
    explicit = [torch.zeros_like(p) for p in model.parameters()]
    for i in range(1000):
        gradients = torch.autograd.grad(nn.functional.cross_entropy(model(x[i:i+1]), y[i:i+1]),
                                        tuple(model.parameters()))
        factor = min(1., C / float(torch.cat([g.flatten() for g in gradients]).norm()))
        for s, g in zip(explicit, gradients):
            s.add_(g * factor)
    full, micro = wrapper(copy.deepcopy(model), C), wrapper(copy.deepcopy(model), C)
    clip_microbatch(full, x, y, C)
    sums = [torch.zeros_like(p) for p in micro.parameters()]
    for xx, yy in zip(x.chunk(10), y.chunk(10)):
        micro.zero_grad(set_to_none=True)
        clip_microbatch(micro, xx, yy, C)
        for s, p in zip(sums, micro.parameters()):
            s.add_(p.grad)
    for s, p, reference in zip(sums, full.parameters(), explicit):
        torch.testing.assert_close(s, p.grad, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(s, reference, atol=5e-5, rtol=1e-5)


def test_local_assets_no_network(monkeypatch):
    import socket
    from torchvision.datasets import CIFAR100
    from exp5.model import pretrained_vit
    from exp5.runtime import ROOT
    def deny_network(*args, **kwargs):
        raise AssertionError('Network access is forbidden')
    monkeypatch.setattr(socket.socket, 'connect', deny_network)
    model, metadata = pretrained_vit()
    assert all(p.requires_grad for p in model.parameters())
    assert (ROOT / metadata['checkpoint_path']).resolve().is_relative_to(ROOT / 'cache')
    assert len(CIFAR100(ROOT / 'data', train=True, download=False)) == 50000
    assert len(CIFAR100(ROOT / 'data', train=False, download=False)) == 10000


class CountingNoise:
    def __init__(self):
        self.calls = 0

    def next(self):
        self.calls += 1
        return [torch.tensor([7., -4.])]

    def marginal_std(self, step):
        return 1.


def test_noise_once_per_logical_step_and_only_dp_momentum():
    p = nn.Parameter(torch.tensor([1., 2.]))
    original = p.detach().clone()
    noise = CountingNoise()
    logical = LogicalSGDM([p], noise, .1)
    for step in range(2):
        old_momentum = logical.momentum[0].clone()
        for i in range(10):
            p.grad = torch.tensor([100., 200.])
            metrics = logical.finish_microbatch()
            if i < 9:
                assert metrics is None and noise.calls == step
                torch.testing.assert_close(logical.momentum[0], old_momentum)
        dp = torch.tensor([1., 2.]) + torch.tensor([7., -4.])
        torch.testing.assert_close(logical.momentum[0], .9 * old_momentum + dp)
        original -= .1 * logical.momentum[0]
        torch.testing.assert_close(p, original)
        assert noise.calls == step + 1


def test_iid_replace_one_sparse_accounting():
    _, _, _, meta = build(METHODS[0], 2.)
    assert meta['per_query_sensitivity'] == 2 * 2. / 1000
    assert meta['sensitivity'] == pytest.approx(2 * 2. * np.sqrt(5) / 1000)
    assert meta['innovation_std'] == pytest.approx(2 * 2. * np.sqrt(5) / (1000 * meta['mu']))
    assert meta['direct_participations'] == 5


def test_band_identity_matches_iid_exactly():
    c_iid, s_iid, _, m_iid = build(METHODS[0], 1.)
    c_mf, s_mf, _, m_mf = build(METHODS[1], 1., num_bands=1)
    np.testing.assert_array_equal(c_iid, c_mf)
    np.testing.assert_array_equal(s_iid, s_mf)
    assert m_iid['innovation_std'] == m_mf['innovation_std']
    p = nn.Parameter(torch.zeros(17))
    a = BandInvMFNoise([p], c_iid, m_iid['innovation_std'], 250, 29)
    b = BandInvMFNoise([p], c_mf, m_mf['innovation_std'], 250, 29)
    for _ in range(10):
        assert torch.equal(a.next()[0], b.next()[0])


def test_band_uses_exp2_momentum_and_sparse_positions():
    from exp2.bandinvmf import build_matrices
    from exp2.privacy import fixed_epoch_sensitivity
    coef, strategy, workload, meta = build(METHODS[1], 1.)
    ref_coef, ref_strategy, ref_workload = build_matrices('momentum_bandinvmf', 250, 4, .9)
    np.testing.assert_allclose(coef, ref_coef)
    np.testing.assert_allclose(strategy, ref_strategy)
    np.testing.assert_allclose(workload, ref_workload)
    assert workload[1] == pytest.approx(1.9)
    assert meta['num_bands'] == 4 and meta['workload'] == 'momentum'
    assert meta['strategy_sensitivity'] == fixed_epoch_sensitivity(strategy, 5, 50)
    np.testing.assert_allclose(materialize(coef, 250) @ strategy, np.eye(250), atol=1e-12)


def test_outputs_and_trial_identity(tmp_path):
    assert tmp_path.resolve().is_relative_to(EXP)
    assert output_path(tmp_path) == tmp_path
    with pytest.raises(ValueError, match='Output must be inside'):
        output_path(EXP.parent / 'exp2/results/forbidden')
    base = Trial(METHODS[0], SEARCH_SEED, 5e-4, 1.)
    assert base.identity == Trial(METHODS[0], SEARCH_SEED, .0005, 1).identity
    assert len({base.identity, Trial(METHODS[1], SEARCH_SEED, 5e-4, 1.).identity,
                Trial(METHODS[0], SEARCH_SEED + 1, 5e-4, 1.).identity,
                Trial(METHODS[0], SEARCH_SEED, 1e-3, 1.).identity,
                Trial(METHODS[0], SEARCH_SEED, 5e-4, 2.).identity}) == 5


def synthetic_summary(trial, top1=.2):
    return dict(status='completed', smoke=False, trial_id=trial.identity, fixed=FIXED,
                **trial.asdict(), optimizer_steps=250, noise_steps=250, physical_batches=2500,
                epochs=[dict(logical_steps=s) for s in (50, 100, 150, 200, 250)],
                final=dict(epsilon=8., test_top1=top1), diagnostics=dict.fromkeys(
                ('clip_fraction', 'mean_unclipped_norm', 'mean_clip_factor', 'query_norm',
                 'noise_std', 'momentum_norm', 'update_norm'), .1), privacy=build(trial.method, trial.C)[3])


def test_staged_grid_dedup_resume_and_freeze(tmp_path):
    executed = []
    stages = []
    def fake_launch(jobs, on_complete):
        stages.append([j['trial'] for j in jobs])
        for job in jobs:
            t = Trial(**job['trial'])
            folder = Path(job['result_dir'])
            if not (folder / 'summary.json').exists():
                folder.mkdir(parents=True)
                # Interior optimum exercises overlapping 3x3 points.
                score = .5 - .01 * abs(np.log2(t.C)) - .01 * abs(np.log2(t.lr / 5e-4))
                (folder / 'summary.json').write_text(json.dumps(synthetic_summary(t, score)))
                executed.append(t.identity)
            on_complete(job)
    selected = StagedSearch(tmp_path, fake_launch).run()
    assert len(stages) == 6 and len(stages[0]) == 5 and len(stages[2]) == 9
    assert [j['C'] for j in stages[0]] == [.25, .5, 1., 2., 4.]
    assert all(j['lr'] == 5e-4 for j in stages[0])
    assert [j['lr'] for j in stages[1]] == [1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3]
    assert all(j['C'] == 1 for j in stages[1])
    assert [j['lr'] for j in stages[4]] == [1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3]
    assert len(executed) == len(set(executed)) == 26
    assert selected['total_completed_search_trials'] == 26
    assert all(selected['methods'][m]['C'] == 1. and selected['methods'][m]['lr'] == 5e-4 for m in METHODS)
    StagedSearch(tmp_path, fake_launch).run()
    assert len(executed) == 26
    assert len(list(tmp_path.glob('stage*_summary.json'))) == 6
    assert (tmp_path / 'iid_frozen.json').exists() and (tmp_path / 'bandinvmf_frozen.json').exists()


def test_selection_ties():
    records = [dict(C=C, lr=lr, final_test_top1=.2) for C, lr in ((1, .01), (.5, .02), (.5, .01))]
    assert winner(records) == records[2]


def test_launcher_three_gpus_one_process_each(monkeypatch, tmp_path):
    from exp5 import launch_batch
    started, live, high_water = [], {}, []
    class Process:
        def __init__(self, command, cwd, env, stdout, stderr):
            self.gpu = env['CUDA_VISIBLE_DEVICES']
            assert self.gpu not in live
            assert command[command.index('--result-dir') + 1].startswith(str(EXP))
            self.pid = len(started) + 1
            self.polls = 0
            live[self.gpu] = self
            started.append(self.gpu)
            high_water.append(len(live))
            folder = Path(command[command.index('--result-dir') + 1])
            (folder / 'summary.json').write_text(json.dumps(dict(status='completed')))
        def poll(self):
            self.polls += 1
            if self.polls < 2:
                return None
            del live[self.gpu]
            return 0
        def terminate(self):
            del live[self.gpu]
        def wait(self):
            return 0
    monkeypatch.setattr(launch_batch.subprocess, 'Popen', Process)
    monkeypatch.setattr(launch_batch.time, 'sleep', lambda _: None)
    jobs = [dict(trial=Trial(METHODS[0], SEARCH_SEED + i, 5e-4, 1.).asdict(),
                 result_dir=str(tmp_path / f'trial_{i}'), smoke=True) for i in range(6)]
    launch_batch.launch(jobs)
    assert started == ['0', '1', '2', '0', '1', '2']
    assert max(high_water) == 3 and not live


def test_final_report_sample_std_and_metadata(tmp_path):
    selected = dict(methods={m: dict(C=1., lr=5e-4) for m in METHODS}, total_completed_search_trials=26)
    rows = [synthetic_summary(Trial(m, seed, 5e-4, 1.), top1=score)
            for m in METHODS for seed, score in zip((20261011, 20261012, 20261013), (.1, .2, .3))]
    result = report(rows, selected, tmp_path)
    for r in result['methods'].values():
        assert r['mean_top1'] == pytest.approx(.2)
        assert r['sample_std_top1'] == pytest.approx(.1)
        assert r['best_seed'] == 20261013 and r['worst_seed'] == 20261011
        assert len(r['seeds']) == 3 and 'privacy' in r['seeds'][0]
