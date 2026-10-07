import copy
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from exp7 import BASE, runtime
from exp7.config import METHODS, SCALE, SEED, cell_for, load_config, trial, trial_id
from exp7 import launcher
from exp2.model import ViTTiny
from exp2.bandinvmf import BandInvMFNoise, build_matrices, materialize
from exp2.privacy import calibrate, epsilon_from_mu
from exp2.scale import ScaledGhostModule, LogicalBatch, clipped_microbatch
runtime()
torch.set_num_threads(2)

def tiny():
    return ViTTiny(patch_size=4, image_size=16, embed_dim=12, depth=1, heads=3, mlp_ratio=2, num_classes=3)

def test_protocol_identity_and_paths():
    cfg = load_config()
    assert cfg['total_steps'] == 250 and cfg['privacy']['k'] == 5 and cfg['privacy']['b_participation'] == 50
    assert all(Path(os.environ[k]).is_relative_to(BASE) for k in ('HF_HOME', 'TORCH_HOME', 'TMPDIR'))
    a = trial(SCALE, .002, 200, .1)
    assert trial_id(a) == trial_id(trial(SCALE, 2e-3, 200., .1))
    with pytest.raises(AssertionError):
        trial(SCALE, .002, 200)
    cfg['seed'] = 20261011
    from exp7.config import validate
    with pytest.raises(AssertionError):
        validate(cfg)

@pytest.mark.parametrize('method', METHODS)
def test_calibration_and_workload(method):
    d, strategy, workload = build_matrices(cell_for(method)['noise'], 250, 4, .9)
    cfg = load_config()
    for C in (1, 30, 2e9):
        cfg['privacy']['max_grad_norm'] = C
        privacy = calibrate(strategy, cfg)
        mu = C * privacy['sensitivity'] / privacy['innovation_std_sum']
        assert epsilon_from_mu(mu, 1e-5) == pytest.approx(8)
    np.testing.assert_allclose(materialize(d, 250) @ strategy, np.eye(250), atol=1e-12)
    if method == SCALE:
        np.testing.assert_array_equal(workload, np.ones(250))
        other = build_matrices('momentum_bandinvmf', 250, 4, .9)[0]
        assert not np.allclose(d, other)

@pytest.mark.parametrize('eps', [None, 1e-8, 1e-4, 1e-3, 1e-2, .1, .3, 1.])
def test_scaled_clipping_matches_explicit_per_example(eps):
    torch.manual_seed(42)
    base = tiny()
    scaled = eps is not None
    C = 20 / eps if scaled else 1.
    wrapped = (ScaledGhostModule(copy.deepcopy(base), C) if scaled else
               GradSampleModuleFastGradientClipping(copy.deepcopy(base), loss_reduction='sum', max_grad_norm=C))
    opt = torch.optim.Adam(wrapped.parameters())
    for p in wrapped.parameters():
        opt.state[p].update(step=torch.tensor(3.), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.rand_like(p) * .001)
    logical = LogicalBatch(wrapped, opt, 4, 1000, noise=None, scaled=scaled, eps_scale=eps or .1)
    logical.begin_microbatch()
    x, y = torch.randn(3, 3, 16, 16), torch.tensor([0, 1, 2])
    sums = [torch.zeros_like(p) for p in base.parameters()]
    norms = []
    for xi, yi in zip(x, y):
        base.zero_grad()
        torch.nn.functional.cross_entropy(base(xi[None]), yi[None]).backward()
        norm = sum(((p.grad * q._logical_scale) if scaled else p.grad).square().sum()
                   for p, q in zip(base.parameters(), wrapped.parameters())).sqrt()
        norms.append(float(norm))
        for total, p in zip(sums, base.parameters()):
            total.add_(p.grad, alpha=min(1., C / float(norm)))
    _, clipped = clipped_microbatch(wrapped, x, y, C)
    assert clipped == sum(n > C for n in norms)
    for p, total in zip(wrapped.parameters(), sums):
        torch.testing.assert_close(p.grad, total, rtol=5e-5, atol=1e-6)

@pytest.mark.parametrize('eps', [1e-8, .1, 1.])
def test_previous_vhat_scale_and_noise_once_per_four_microbatches(eps):
    model = tiny()
    opt = torch.optim.Adam(model.parameters(), lr=.002, betas=(.9, .999), eps=1e-8)
    d, _, _ = build_matrices('prefix_bandinvmf', 3, 4, .9)
    noise = BandInvMFNoise(model.parameters(), d, .1, 3, 123)
    logical = LogicalBatch(model, opt, 4, 1000, noise, scaled=True, eps_scale=eps)
    for step in range(3):
        previous = [torch.zeros_like(p) if step == 0 else opt.state[p]['exp_avg_sq'] / (1 - .999**step) for p in model.parameters()]
        for micro in range(4):
            logical.begin_microbatch()
            for p, vhat in zip(model.parameters(), previous):
                torch.testing.assert_close(p._logical_scale, 1 / (vhat.sqrt() + eps))
                p.grad = torch.ones_like(p) * (step + 1)
            done = logical.finish_microbatch()
            assert done == (micro == 3)
            assert logical.optimizer_steps == noise.step_count == step + int(done)


def test_fifo_three_gpu_dedup_reuse_and_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'BASE', tmp_path)
    (tmp_path / 'results').mkdir()
    script = tmp_path / 'worker.py'
    script.write_text('''import json, os, sys, time
from pathlib import Path
j=json.loads(sys.argv[1]); d=Path(sys.argv[2]); d.mkdir(parents=True)
time.sleep(.15)
j.update(status='completed', smoke=True, final_test_top1=.1, gpu=os.environ['CUDA_VISIBLE_DEVICES'])
(d/'summary.json').write_text(json.dumps(j))
''')
    jobs = [trial('dp-adam-iid', lr, 30) for lr in (.001, .002, .003, .004)]
    def command(j, d, gpu):
        return [sys.executable, '-B', str(script), json.dumps(j), str(d)]
    rows = launcher.run_queue(jobs + jobs[:1], [0, 1, 2], smoke=True, command=command, poll_seconds=.02)
    assert len(rows) == 4
    events = [json.loads(s) for s in (tmp_path / 'results/scheduler.jsonl').read_text().splitlines()]
    starts = [r for r in events if r['event'] == 'start']
    assert [r['lr'] for r in starts] == [.001, .002, .003, .004]
    live = set()
    for r in events:
        if r['event'] == 'start':
            assert r['gpu'] not in live
            live.add(r['gpu'])
        else:
            live.remove(r['gpu'])
        assert len(live) <= 3
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **k: pytest.fail('must reuse'))
    assert len(launcher.run_queue(jobs, [0, 1, 2], smoke=True)) == 4
    extra = trial('dp-adam-iid', .01, 30)
    (tmp_path / 'results/smoke' / trial_id(extra)).mkdir()
    with pytest.raises(RuntimeError, match='Incomplete'):
        launcher.run_queue([extra], [0, 1, 2], smoke=True)
