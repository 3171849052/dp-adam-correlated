import copy
import json
from pathlib import Path
import socket
import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from exp3 import model as models
from exp3 import train, launch_batch
from exp3.bandinvmf import BandInvMFNoise, build_matrices, materialize
from exp3.diagnostics import probe_gain, statistics, MuonDiagnostics
from exp3.final_runner import final_specs, FINAL_SEEDS, verify_pairing
from exp3.geometry import Identity, NormScale, SpectralScale, freeze_geometry
from exp3.mechanism import GeometryGhostModule, LogicalBatch, clipped_microbatch
from exp3.optimizer import HybridOptimizer, muon_map, partition
from exp3.privacy import calibrate, epsilon_from_mu
from exp3.spec import TrialSpec, EXP3, ROOT, METHODS, TRIAL_FILES, read_completed


def tiny():
    return models.ViTTiny(patch_size=4, embed_dim=8, depth=1, heads=2,
                          mlp_ratio=2, num_classes=3, image_size=8)


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_fixed_config_and_dataset_local_only(monkeypatch):
    calls = []
    class Data:
        def __init__(self, root, train, transform, download):
            calls.append((root, train, download))
            self.size = 50000 if train else 10000
        def __len__(self):
            return self.size
    monkeypatch.setattr(train.datasets, 'CIFAR100', Data)
    train.load_data(dict(mean=[.5]*3, std=[.5]*3, crop_pct=.9))
    assert calls == [(ROOT/'data', True, False), (ROOT/'data', False, False)]
    c = yaml.safe_load((EXP3 / 'config.yaml').read_text())
    assert (c['physical_batch_size'], c['logical_batch_size'], c['gradient_accumulation']) == (250, 1000, 4)
    assert c['privacy']['k'] == 5 and c['privacy']['b_participation'] == 50
    assert c['epochs'] == 5 and c['total_steps'] == 250
    assert c['privacy']['sampling_amplification'] is False


def test_missing_checkpoint_fails_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(models, 'CHECKPOINT_CACHE', tmp_path)
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('Network requested'))
    with pytest.raises(FileNotFoundError):
        models.pretrained_vit()


def test_pretrained_partition_and_paired_initialization(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('Network requested'))
    digests, head_digests, aug = [], [], []
    image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (32,32,3), dtype=np.uint8))
    for method in METHODS:
        train.seed_all(20261011)
        model, metadata = models.pretrained_vit()
        assert models.checkpoint_path().resolve().is_relative_to(ROOT / 'cache')
        assert metadata['checkpoint_sha256'] == models.checkpoint_sha256(models.checkpoint_path())
        muon, adam = partition(model)
        assert len(muon) == 48
        assert {'patch.weight', 'patch.bias', 'cls.weight', 'position.weight', 'head.weight', 'head.bias'} <= set(adam)
        assert not {id(p) for p in muon.values()} & {id(p) for p in adam.values()}
        assert set(muon) | set(adam) == dict(model.named_parameters()).keys()
        assert all(p.requires_grad for p in model.parameters())
        digests.append(models.initialization_digest(model))
        head_digests.append(models.initialization_digest(model.head))
        aug.append(train.image_transforms(metadata)[0](image))
    assert len(set(digests)) == len(set(head_digests)) == 1
    for x in aug[1:]:
        torch.testing.assert_close(x, aug[0], rtol=0, atol=0)
    np.testing.assert_array_equal(train.training_order(20261011), train.training_order(20261011))
    assert not torch.equal(train.training_order(20261011), train.training_order(20261012))


def reference_phi(h):
    x = h.T if h.shape[0] > h.shape[1] else h.clone()
    x = x / (torch.linalg.vector_norm(x) + 1e-7)
    for _ in range(5):
        a = x @ x.T
        x = 3.4445*x - 4.7750*a@x + 2.0315*a@a@x
    return x.T if h.shape[0] > h.shape[1] else x


def test_muon_nesterov_pre_ns_and_independent_adam_lr():
    torch.manual_seed(0)
    model = tiny()
    opt = HybridOptimizer(model, .02, .0003)
    expected_adam_model = copy.deepcopy(model)
    _, adam = partition(expected_adam_model)
    expected_adam = torch.optim.Adam(adam.values(), lr=.0003, betas=(.9,.999), eps=1e-8, weight_decay=0)
    buffers = {n: torch.zeros_like(p) for n,p in opt.muon.items()}
    for _ in range(3):
        expected_muon = {}
        for name, p in model.named_parameters():
            p.grad = torch.randn_like(p)
            if name in opt.muon:
                buffers[name] = .95*buffers[name] + .05*p.grad
                h = .05*p.grad + .95*buffers[name]
                expected_muon[name] = (h, p.detach().clone() - .02*reference_phi(h)*max(1,p.shape[0]/p.shape[1])**.5)
            else:
                adam[name].grad = p.grad.clone()
        expected_adam.step()
        opt.step()
        for name,(h,pnew) in expected_muon.items():
            torch.testing.assert_close(opt.state[opt.muon[name]]['pre_ns'], h)
            torch.testing.assert_close(opt.muon[name], pnew, atol=2e-6, rtol=2e-5)
        for name,p in opt.adam.items():
            torch.testing.assert_close(p, adam[name])
    assert opt.param_groups[0]['lr'] == .02 and opt.param_groups[1]['lr'] == .0003
    assert opt.param_groups[0]['momentum'] == .95 and opt.param_groups[0]['nesterov']
    assert opt.param_groups[0]['ns_steps'] == 5


@pytest.mark.parametrize('shape', [(3,5),(5,3),(4,4)])
def test_normscale_inverse_and_identity(shape):
    torch.manual_seed(3)
    h,g = torch.randn(shape, dtype=torch.double), torch.randn((2,)+shape, dtype=torch.double)
    s = NormScale(h, .2)
    torch.testing.assert_close(s.inverse(s.transform(g)), g)
    torch.testing.assert_close(NormScale(h,1).transform(g), g, rtol=0, atol=0)
    with pytest.raises(AssertionError):
        NormScale(h, 0)


@pytest.mark.parametrize('shape,rank', [((3,5),3),((5,3),1),((4,4),2)])
def test_spectral_inverse_range_normalization(shape, rank):
    torch.manual_seed(4)
    h = torch.randn(shape[0],rank,dtype=torch.double)@torch.randn(rank,shape[1],dtype=torch.double)
    g = torch.randn((2,)+shape,dtype=torch.double)
    s = SpectralScale(h,4,.1)
    torch.testing.assert_close(s.inverse(s.transform(g)),g)
    assert s.delta > 0 and s.pairwise_min >= .25-1e-12 and s.pairwise_max <= 4+1e-12
    assert float(torch.quantile(s.active_gain.log(),.5)) == pytest.approx(0, abs=1e-12)
    one = SpectralScale(h,1,.1)
    torch.testing.assert_close(one.transform(g),g)


@pytest.mark.parametrize('method', METHODS)
def test_first_step_identity_and_only_muon_geometry(method):
    opt = HybridOptimizer(tiny(), .01, .001)
    geometries = freeze_geometry(opt, method, .2, 4, .1)
    assert all(isinstance(s,Identity) for s in geometries.values())
    for p in opt.muon.values():
        opt.state[p]['pre_ns'] = torch.randn_like(p)
    geometries = freeze_geometry(opt, method, .2, 4, .1)
    assert all(isinstance(geometries[p],Identity) for p in opt.adam.values())
    assert all(isinstance(s,Identity) for s in freeze_geometry(opt,method,1,1,.1).values())


@pytest.mark.parametrize('method', ['iid_dp_hybrid','mf_muon_standard','mf_muon_normscale','mf_muon_spectralscale'])
def test_joint_global_clip_matches_per_example_autograd(method):
    torch.manual_seed(7)
    reference = tiny()
    model = copy.deepcopy(reference)
    opt = HybridOptimizer(model,.01,.001)
    for p in opt.muon.values():
        opt.state[p]['pre_ns'] = torch.randn_like(p)
    wrapped = GeometryGhostModule(model, 1.)
    logical = LogicalBatch(wrapped,opt,None,method,.7,.2,3,.1)
    logical.begin_microbatch()
    x,y = torch.randn(3,3,8,8),torch.tensor([0,1,2])
    sums = [torch.zeros_like(p) for p in reference.parameters()]
    norms=[]
    for xi, yi in zip(x,y):
        reference.zero_grad()
        torch.nn.functional.cross_entropy(reference(xi[None]),yi[None]).backward()
        norm = sum(logical.geometries[p].transform(r.grad).square().sum()
                   for p,r in zip(model.parameters(), reference.parameters())).sqrt()
        norms.append(float(norm))
        for total,p in zip(sums, reference.parameters()):
            total.add_(p.grad,alpha=min(1.,.7/float(norm)))
    _, clipped = clipped_microbatch(wrapped,x,y,.7)
    assert clipped == sum(n>.7 for n in norms)
    for p,total in zip(model.parameters(),sums):
        torch.testing.assert_close(p.grad,total,atol=2e-6,rtol=5e-5)


@pytest.mark.parametrize('method',METHODS)
def test_accumulation_one_joint_noise_and_optimizer_step(method):
    torch.manual_seed(2)
    model = tiny()
    opt = HybridOptimizer(model,.01,.001)
    private = method != 'nonprivate_hybrid'
    class CountingNoise:
        def __init__(self): self.step_count=0
        def next(self):
            self.step_count+=1
            return [torch.ones_like(p)*.03 for p in model.parameters()]
    noise = CountingNoise() if private else None
    logical = LogicalBatch(model,opt,noise,method,1,.2,4,.1)
    for step in range(2):
        for micro in range(4):
            logical.begin_microbatch()
            if micro==0:
                frozen=logical.geometries
            assert logical.geometries is frozen
            for p in model.parameters(): p.grad=torch.ones_like(p)
            expected = [s.inverse(s.transform(torch.ones_like(p)*4)+torch.ones_like(p)*.03)/1000
                        if private else torch.ones_like(p)*4/1000 for p,s in frozen.items()]
            # dictionary iteration is Muon then Adam, compare by parameter keys below.
            completed = logical.finish_microbatch()
            assert completed == (micro==3)
            assert opt.step_count == logical.optimizer_steps == step+int(completed)
            if private: assert noise.step_count == opt.step_count
            if completed:
                for p,want in zip(frozen,expected): torch.testing.assert_close(p.grad,want)
    assert opt.step_count==2


def test_mf_workload_factorization_and_clip_recalibration():
    built=[build_matrices('momentum_bandinvmf',250,4,.9) for _ in range(3)]
    for current in built[1:]:
        for a,b in zip(current,built[0]): np.testing.assert_array_equal(a,b)
    d,c,w = built[0]
    np.testing.assert_allclose(w,(1-.9**np.arange(1,251))/.1,rtol=1e-14)
    np.testing.assert_allclose(materialize(d,250)@c,np.eye(250),atol=1e-12)
    cfg=yaml.safe_load((EXP3/'config.yaml').read_text())
    cfg['privacy']['max_grad_norm']=1
    p1=calibrate(c,cfg)
    cfg['privacy']['max_grad_norm']=100
    p2=calibrate(c,cfg)
    assert p2['innovation_std_sum']==pytest.approx(100*p1['innovation_std_sum'])
    assert epsilon_from_mu(p2['target_mu'],1e-5)==pytest.approx(8)
    for c in (np.eye(250),built[0][1]):
        p=calibrate(c,cfg)
        assert epsilon_from_mu(100*p['sensitivity']/p['innovation_std_sum'],1e-5)==pytest.approx(8)


def test_noise_innovation_convolution():
    d,_,_=build_matrices('momentum_bandinvmf',8,4,.9)
    parameters=[torch.nn.Parameter(torch.zeros(2,3)),torch.nn.Parameter(torch.zeros(2))]
    noise=BandInvMFNoise(parameters,d,1.3,8,71)
    rng=torch.Generator().manual_seed(71)
    z=np.stack([torch.cat([(torch.randn(p.shape,generator=rng)*1.3).flatten() for p in parameters]).numpy() for _ in range(8)])
    outputs=np.stack([torch.cat([e.flatten() for e in noise.next()]).numpy() for _ in range(8)])
    np.testing.assert_allclose(outputs,materialize(d,8)@z,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('geometry',['identity','norm','spectral'])
def test_jvp_matches_finite_difference(geometry):
    torch.manual_seed(5)
    h,e=torch.randn(5,3,dtype=torch.double),torch.randn(5,3,dtype=torch.double)
    s={'identity':Identity(), 'norm':NormScale(h,.3), 'spectral':SpectralScale(h,4,.1)}[geometry]
    v=s.inverse(e)
    fd=(muon_map(h+1e-6*v)-muon_map(h-1e-6*v))/(2e-6)
    assert probe_gain(h,e,s)==pytest.approx(float(fd.norm()/e.norm()),rel=2e-6)
    stats=statistics([1,2,3,4])
    assert stats['cv']==pytest.approx(np.std([1,2,3,4])/2.5)


def test_diagnostic_fixed_probes_and_temporal_statistics(tmp_path):
    model=tiny(); opt=HybridOptimizer(model,.01,.001)
    name='blocks.0.attn.qkv.weight'
    a=MuonDiagnostics(opt,interval=2,probes=3,layers=(name,))
    b=MuonDiagnostics(opt,interval=2,probes=3,layers=(name,))
    for x,y in zip(a.probes[name],b.probes[name]): torch.testing.assert_close(x,y,rtol=0,atol=0)
    for step in range(1,4):
        for p in model.parameters(): p.grad=torch.ones_like(p)*(step*.2)
        geometry=freeze_geometry(opt,'mf_muon_normscale',.3,4,.1)
        opt.step(); a.record(step,geometry)
    result=a.save(tmp_path)
    assert [r['step'] for r in result['records']]==[1,2]
    assert 'temporal_update_gain_cv' in result and 'layer_update_gain_cv' in result


def completed_fixture(spec):
    spec.directory.mkdir(parents=True)
    for name in TRIAL_FILES: (spec.directory/name).write_text('nonempty')
    summary=dict(status='completed',spec_sha256=spec.fingerprint())
    (spec.directory/'summary.json').write_text(json.dumps(summary))
    return summary


def test_reuse_and_incomplete_rejected(tmp_path):
    spec=TrialSpec('mf_muon_standard',str(tmp_path/'trial'))
    spec.directory.mkdir()
    with pytest.raises(RuntimeError,match='Incomplete'): read_completed(spec)
    for name in TRIAL_FILES: (spec.directory/name).write_text('nonempty')
    (spec.directory/'summary.json').write_text(json.dumps(dict(status='completed',spec_sha256=spec.fingerprint())))
    assert read_completed(spec)['status']=='completed'
    different=TrialSpec('mf_muon_standard',str(spec.directory),muon_lr=.1)
    with pytest.raises(RuntimeError,match='conflicts'): read_completed(different)
    with pytest.raises(AssertionError): TrialSpec('mf_muon_standard',str(ROOT/'exp2/new'))


def test_launcher_gpu_limit_refill_failure_and_reuse(tmp_path,monkeypatch):
    active, launched, finished={},[],[]
    jobs=[TrialSpec('mf_muon_standard',str(tmp_path/f'trial_{i}')) for i in range(5)]
    class Process:
        def __init__(self,cmd,cwd,env,stdout,stderr):
            gpu=int(env['CUDA_VISIBLE_DEVICES'])
            assert gpu in (1,2,3) and gpu not in active
            assert stderr==launch_batch.subprocess.STDOUT and stdout.name.endswith('train.log')
            assert cmd[cmd.index('-m')+1]=='exp3.run_trial'
            self.gpu,self.polls=gpu,0; self.fail=len(launched)==1
            active[gpu]=self; launched.append((gpu,len(finished)))
        def poll(self):
            self.polls+=1
            if self.polls<(2 if self.gpu==2 else 4): return None
            del active[self.gpu]; finished.append(self.gpu)
            return 7 if self.fail else 0
    monkeypatch.setattr(launch_batch.subprocess,'Popen',Process)
    monkeypatch.setattr(launch_batch.time,'sleep',lambda _:None)
    monkeypatch.setattr(launch_batch,'read_completed',lambda _:dict(status='completed'))
    summary=launch_batch.run_queue(jobs,tmp_path/'batch')
    assert summary['status']=='failed' and len(summary['trials'])==5
    assert launched[3]==(2,1) and not active
    assert json.loads((tmp_path/'batch/batch_summary.json').read_text())['status']=='failed'
    monkeypatch.setattr(launch_batch.subprocess,'Popen',lambda *a,**k:pytest.fail('reused trial relaunched'))
    assert all(r['reused'] for r in launch_batch.run_queue(jobs,tmp_path/'reused')['trials'])


def test_final_fifteen_specs_and_pairing(tmp_path):
    chosen=json.loads((EXP3/'specs/frozen.example.json').read_text())
    specs=final_specs(chosen,tmp_path/'final')
    assert len(specs)==15 and {s.seed for s in specs}==set(FINAL_SEEDS)
    rows=[]
    for s in specs:
        s.directory.mkdir(parents=True)
        np.save(s.directory/'train_order.npy',train.training_order(s.seed).numpy())
        summary=dict(checkpoint_sha256='same',initialization_sha256=f'init{s.seed}',
                     classifier_initialization_sha256=f'head{s.seed}',train_order_sha256=f'order{s.seed}',
                     augmentation_trace_sha256=[f'aug{s.seed}'])
        rows.append(dict(spec=s.mapping(),summary=summary))
    verify_pairing(rows)
    rows[1]['summary']['initialization_sha256']='wrong'
    with pytest.raises(AssertionError,match='Pairing mismatch'): verify_pairing(rows)


def test_cli_and_json_numeric_spec_fingerprints_match(tmp_path):
    json_spec=TrialSpec('mf_muon_standard',str(tmp_path/'trial'),max_grad_norm=100,kappa=4,lambda_parallel=1)
    cli_spec=TrialSpec('mf_muon_standard',str(tmp_path/'trial'),max_grad_norm=100.,kappa=4.,lambda_parallel=1.)
    assert json_spec.fingerprint()==cli_spec.fingerprint()


def test_launcher_incomplete_preflight_fails_without_launching(tmp_path,monkeypatch):
    invalid=TrialSpec('mf_muon_standard',str(tmp_path/'invalid'))
    invalid.directory.mkdir()
    (invalid.directory/'train.log').write_text('interrupted')
    pending=TrialSpec('iid_dp_hybrid',str(tmp_path/'pending'))
    monkeypatch.setattr(launch_batch.subprocess,'Popen',lambda *a,**k:pytest.fail('Invalid batch launched'))
    summary=launch_batch.run_queue([invalid,pending],tmp_path/'output')
    assert summary['status']=='failed'
    assert 'Incomplete trial' in summary['trials'][0]['error']
    assert not pending.directory.exists()
    assert (invalid.directory/'train.log').read_text()=='interrupted'
    assert (tmp_path/'output/batch_summary.json').is_file()


def test_batch_cli_input_count_and_nonzero_failure(tmp_path,monkeypatch):
    import sys
    specs=tmp_path/'specs.json'
    supplied=[TrialSpec('mf_muon_standard',str(tmp_path/'trial')).mapping()]
    specs.write_text(json.dumps(supplied))
    monkeypatch.setattr(sys,'argv',['exp3.launch_batch','--specs',str(specs)])
    monkeypatch.setattr(launch_batch,'run_queue',lambda *a:dict(status='failed',trials=[dict(returncode=7)]))
    with pytest.raises(SystemExit) as failure: launch_batch.main()
    assert failure.value.code==1
    specs.write_text(json.dumps(supplied*4))
    with pytest.raises(AssertionError): launch_batch.main()


def test_iid_uses_common_momentum_workload():
    iid=build_matrices('iid',250,4,.9)
    mf=build_matrices('momentum_bandinvmf',250,4,.9)
    np.testing.assert_array_equal(iid[2],mf[2])
    np.testing.assert_array_equal(iid[0],[1.])
    np.testing.assert_array_equal(iid[1],np.eye(250))


def test_final_runner_frozen_snapshot_aggregation_and_change_rejection(tmp_path,monkeypatch):
    import sys
    from exp3 import final_runner
    frozen=tmp_path/'frozen.json'
    frozen.write_bytes((EXP3/'specs/frozen.example.json').read_bytes())
    output=tmp_path/'final'
    def fake_queue(specs,directory):
        assert len(specs)==15 and all(not s.smoke for s in specs)
        rows=[]
        for spec in specs:
            spec.directory.mkdir(parents=True)
            np.save(spec.directory/'train_order.npy',np.arange(5))
            summary=dict(checkpoint_sha256='checkpoint',initialization_sha256=f'init_{spec.seed}',
                         classifier_initialization_sha256=f'head_{spec.seed}',train_order_sha256='order',
                         augmentation_trace_sha256=[f'aug_{spec.seed}'],
                         final_test_top1=.2+.01*METHODS.index(spec.method)+.001*FINAL_SEEDS.index(spec.seed))
            rows.append(dict(spec=spec.mapping(),summary=summary,returncode=0))
        return dict(status='completed',trials=list(reversed(rows)))
    monkeypatch.setattr(final_runner,'run_queue',fake_queue)
    monkeypatch.setattr(sys,'argv',['exp3.final_runner','--frozen-config',str(frozen),'--result-dir',str(output)])
    final_runner.main()
    result=json.loads((output/'final_summary.json').read_text())
    assert result['trials']==15 and result['paired']
    assert result['methods']['mf_muon_standard']['final_test_top1_mean']==pytest.approx(.221)
    assert result['methods']['mf_muon_standard']['final_test_top1_sample_std']==pytest.approx(.001)
    assert result['methods']['mf_muon_standard']['per_seed']['20261011']==pytest.approx(.22)
    assert (output/'frozen_config.json').read_bytes()==frozen.read_bytes()
    frozen.write_text(frozen.read_text()+'\n')
    with pytest.raises(AssertionError,match='Frozen configuration changed'): final_runner.main()
