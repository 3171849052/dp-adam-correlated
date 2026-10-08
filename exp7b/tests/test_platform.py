import copy
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from exp7b import BASE, runtime
from exp7b import config, launcher, audit
from exp7b.config import *
from exp7b.bandinvmf import *
from exp2.model import ViTTiny
from exp2.privacy import calibrate, epsilon_from_mu
from exp2.scale import ScaledGhostModule, LogicalBatch, clipped_microbatch
runtime()
torch.set_num_threads(2)

def tiny():
    return ViTTiny(patch_size=4,image_size=16,embed_dim=12,depth=1,heads=3,mlp_ratio=2,num_classes=3)

def test_protocol_and_no_external_writes():
    cfg = load_config()
    assert (cfg['total_steps'], cfg['privacy']['k'],cfg['privacy']['b_participation']) == (250,5,50)
    assert all(Path(os.environ[k]).is_relative_to(BASE) for k in ('HF_HOME','TORCH_HOME','TMPDIR','XDG_CACHE_HOME'))
    assert trial_dir(trial(SGD,.002,10),'search') != trial_dir(trial(SGD,.002,10),'smoke')
    with pytest.raises(AssertionError):
        trial(MOMENTUM_SCALE,.005,100)
    cfg['total_steps'] = 251
    with pytest.raises(AssertionError):
        validate(cfg)

def test_bias_workload_exact_non_toeplitz():
    n, beta = 8, .9
    W = momentum_bias_workload(n,beta)
    expected = np.zeros((n,n))
    for t in range(1,n+1):
        for j in range(1,t+1):
            expected[t-1,j-1] = sum(beta**(k-j)/(1-beta**k) for k in range(j,t+1))
    np.testing.assert_allclose(W,expected,rtol=1e-14)
    assert not np.isclose(W[0,0],W[1,1])

@pytest.mark.parametrize('method',METHODS)
def test_matrices_and_privacy(method):
    d,S,W = build_matrices(cell_for(method)['noise'],250,4,.9)
    D = materialize(d,250)
    np.testing.assert_allclose(D @ S,np.eye(250),atol=2e-12)
    assert W.shape == S.shape == (250,250) and np.isfinite(S).all()
    cfg = load_config()
    for C in (1,30,1000):
        cfg['privacy']['max_grad_norm'] = C
        p = calibrate(S,cfg)
        assert epsilon_from_mu(C*p['sensitivity']/p['innovation_std_sum'],1e-5) == pytest.approx(8)
    if method != IID:
        assert len(d) == 4
    if method.endswith('-scale'):
        other = build_matrices(cell_for(method[:-6])['noise'],250,4,.9)
        for a,b in zip((d,S,W),other):
            np.testing.assert_array_equal(a,b)

def test_bias_objective_uses_full_matrix():
    d,S,W,meta = bias_matrices()
    assert meta['converged'] and meta['full_workload_shape'] == [250,250]
    direct = fixed_epoch_sensitivity(S,5,50)**2 * np.square(W @ materialize(d,250)).sum()/250
    assert direct == pytest.approx(meta['optimized_objective'],rel=1e-11)
    assert direct <= meta['initial_objective']
    # Substituting the first column as a Toeplitz workload changes the error.
    surrogate = materialize(W[:,0],250)
    assert not np.isclose(workload_error(W,d),workload_error(surrogate,d),rtol=.01)

@pytest.mark.parametrize('eps',[None,.03,.1,.3,1.])
def test_clipping_matches_explicit_gradients(eps):
    torch.manual_seed(42)
    base = tiny()
    C = 10/eps if eps else 1.
    model = ScaledGhostModule(copy.deepcopy(base),C) if eps else GradSampleModuleFastGradientClipping(copy.deepcopy(base),loss_reduction='sum',max_grad_norm=C)
    opt = torch.optim.Adam(model.parameters())
    for p in model.parameters():
        opt.state[p].update(step=torch.tensor(3.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.rand_like(p)*.001)
    logical = LogicalBatch(model,opt,4,1000,None,scaled=eps is not None,eps_scale=eps or .1)
    logical.begin_microbatch()
    x,y = torch.randn(3,3,16,16),torch.tensor([0,1,2])
    totals = [torch.zeros_like(p) for p in base.parameters()]
    norms = []
    for xi,yi in zip(x,y):
        base.zero_grad()
        torch.nn.functional.cross_entropy(base(xi[None]),yi[None]).backward()
        norm = sum((p.grad*q._logical_scale if eps else p.grad).square().sum() for p,q in zip(base.parameters(),model.parameters())).sqrt()
        norms.append(float(norm))
        for total,p in zip(totals,base.parameters()):
            total.add_(p.grad,alpha=min(1.,C/float(norm)))
    _,clipped = clipped_microbatch(model,x,y,C)
    assert clipped == sum(n>C for n in norms)
    for p,total in zip(model.parameters(),totals):
        torch.testing.assert_close(p.grad,total,rtol=5e-5,atol=1e-6)

def test_previous_vhat_and_inverse_scale_noise():
    model = tiny()
    opt = torch.optim.Adam(model.parameters(),lr=.002,eps=1e-8)
    class Noise:
        step_count = 0
        def next(self):
            self.step_count += 1
            return [torch.full_like(p,2.) for p in model.parameters()]
    noise = Noise()
    logical = LogicalBatch(model,opt,4,1000,noise,scaled=True,eps_scale=.1)
    for step in range(3):
        previous = [torch.zeros_like(p) if step == 0 else opt.state[p]['exp_avg_sq']/(1-.999**step) for p in model.parameters()]
        for micro in range(4):
            logical.begin_microbatch()
            scales = [p._logical_scale.clone() for p in model.parameters()]
            for p,v in zip(model.parameters(),previous):
                torch.testing.assert_close(p._logical_scale,1/(v.sqrt()+.1))
                p.grad = torch.ones_like(p)
            assert logical.finish_microbatch() == (micro==3)
            assert logical.optimizer_steps == noise.step_count == step+int(micro==3)
        for p,s in zip(model.parameters(),scales):
            torch.testing.assert_close(p.grad,(4+2/s)/1000)

def test_fifo_reuse_and_gpu_zero_excluded(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'BASE',tmp_path)
    monkeypatch.setattr(launcher,'BASE',tmp_path)
    monkeypatch.setattr(audit,'BASE',tmp_path)
    worker = tmp_path/'worker.py'
    worker.write_text("""import json,os,sys,time
from pathlib import Path
j=json.loads(sys.argv[1]);d=Path(sys.argv[2]);d.mkdir(parents=True)
time.sleep(.1)
j.update(status='completed',smoke=True,final_test_top1=.1,gpu=os.environ['CUDA_VISIBLE_DEVICES'])
(d/'summary.json').write_text(json.dumps(j))
""")
    jobs = [trial(IID,lr,30) for lr in (.001,.002,.003,.004)]
    command = lambda j,d,g: [sys.executable,'-B',str(worker),json.dumps(j),str(d)]
    results = launcher.run_queue(jobs+jobs[:1],[1,2,3],category='smoke',command=command,poll_seconds=.02)
    assert len(results) == 4
    events = [json.loads(s) for s in (tmp_path/'results/scheduler.jsonl').read_text().splitlines()]
    assert [e['lr'] for e in events if e['event']=='start'] == [.001,.002,.003,.004]
    assert audit.audit_fifo()['status'] == 'passed'
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**k:pytest.fail('must reuse'))
    assert len(launcher.run_queue(jobs,[1,2,3],category='smoke',command=command)) == 4
    with pytest.raises(AssertionError):
        launcher.run_queue(jobs,[0,1,2],category='smoke')

def test_accuracy_selection_and_t_statistics():
    from exp7b.report import best,statistics
    a = dict(trial(SGD,.001,10),status='completed',smoke=False,final_test_top1=.73,train_loss=.1)
    b = dict(trial(SGD,.002,10),status='completed',smoke=False,final_test_top1=.74,train_loss=9)
    failure = dict(a,status='numerical_failure',final_test_top1=None)
    assert best([a,b,failure]) is b
    values = np.linspace(.7,.79,10)
    st = statistics(values)
    assert st['mean'] == pytest.approx(.745)
    assert st['sample_std'] == pytest.approx(values.std(ddof=1))
    assert (st['ci95'][1]-st['mean'])/st['standard_error'] == pytest.approx(2.2621571628)

def test_freeze_immutable_and_70_paired_jobs(tmp_path,monkeypatch):
    from exp7b import frozen,final
    monkeypatch.setattr(config,'BASE',tmp_path)
    monkeypatch.setattr(frozen,'BASE',tmp_path)
    fixed = frozen.import_fixed()
    assert all(r['source']=='exp7_frozen' for r in fixed.values())
    selected = {m:dict(trial(m,.005,100,.1 if m.endswith('-scale') else None),source='search_selected') for m in SEARCH_METHODS}
    frozen.freeze(selected)
    jobs = final.final_jobs()
    assert len(jobs) == 70 and len({trial_id(j) for j in jobs}) == 70
    for seed in FINAL_SEEDS:
        assert {j['method'] for j in jobs if j['seed']==seed} == set(METHODS)
    selected[SGD]['lr'] = .1
    with pytest.raises(AssertionError,match='immutable'):
        frozen.freeze(selected)
    path = tmp_path/'results/frozen_configs.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(AssertionError):
        frozen.load_frozen()

def test_pairing_detects_rng_mismatch():
    keys = ('initialization_sha256','classifier_initialization_sha256','pretrained_checkpoint_sha256','train_order_sha256','augmentation_trace_sha256')
    a = dict(seed=20261011,**{k:'same' for k in keys})
    b = dict(a,augmentation_trace_sha256='different')
    with pytest.raises(AssertionError):
        audit.audit_pairing([a,b])
