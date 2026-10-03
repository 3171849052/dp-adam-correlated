import copy
import csv
import json
import socket
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from safetensors.torch import load_file
from exp2 import model as models
from exp2.bandinvmf import BandInvMFNoise, build_matrices, materialize
from exp2.cells import (CELLS, ANCHORS, CLIPS, LR_GRIDS, FINAL_SEEDS, SEARCH_SEED,
                        first_search_trials, second_search_trials, final_trials)
from exp2.diagnostics import (coordinate_indices, MechanismTrace, mf_distortion,
                              factorial_effects, mechanism_metrics)
from exp2.privacy import calibrate, gdp_delta
from exp2.scale import LogicalBatch, ScaledGhostModule, clipped_microbatch
from exp2.train import ROOT, load_config, seed_all, image_transforms
from exp2 import sweep

CONFIG = load_config(ROOT / 'exp2/config.yaml')
torch.set_num_threads(2)


def tiny():
    return models.ViTTiny(patch_size=4, image_size=16, embed_dim=24, depth=1,
                          heads=3, mlp_ratio=2, num_classes=3)


def configs():
    return dict(ANCHORS, iid_scale=dict(lr=.002, max_grad_norm=200., eps_scale=.1),
                prefix_standard=dict(lr=.002, max_grad_norm=1., num_bands=4),
                momentum_scale=dict(lr=.002, max_grad_norm=200., eps_scale=.1, num_bands=4))


def test_fixed_config_and_six_cells():
    assert (CONFIG['physical_batch_size'], CONFIG['logical_batch_size'], CONFIG['gradient_accumulation']) == (250, 1000, 4)
    assert (CONFIG['privacy']['k'], CONFIG['privacy']['b_participation'], CONFIG['total_steps']) == (5, 50, 250)
    raw = yaml.safe_load((ROOT / 'exp2/config.yaml').read_text())
    assert 'gradient_accumulation' not in raw and 'total_steps' not in raw
    assert 'k' not in raw['privacy'] and 'b_participation' not in raw['privacy']
    assert len(CELLS) == 6
    assert {(v['geometry'], v['noise']) for v in CELLS.values()} == {
        (g, n) for g in ('standard', 'scale') for n in ('iid', 'prefix_bandinvmf', 'momentum_bandinvmf')}
    assert ANCHORS == {'iid_standard': dict(lr=5e-4, max_grad_norm=1.),
                       'prefix_scale': dict(lr=2e-3, max_grad_norm=200., eps_scale=.1, num_bands=4),
                       'momentum_standard': dict(lr=5e-3, max_grad_norm=1., num_bands=4)}


def test_search_grid_and_no_duplicate_anchor_runs():
    first = first_search_trials()
    second = second_search_trials({'iid_scale': 100., 'momentum_scale': 300.})
    assert len(first) == 16 and len(second) == 7
    assert all(j['seed'] == SEARCH_SEED for j in first + second)
    assert not (set(j['method'] for j in first + second) & set(ANCHORS))
    for method in ('iid_scale', 'momentum_scale'):
        clips = [j for j in first if j['method'] == method]
        assert tuple(j['max_grad_norm'] for j in clips) == CLIPS == (50, 100, 200, 300, 500)
        assert all(j['lr'] == .002 and j['eps_scale'] == .1 for j in clips)
        assert {j['lr'] for j in second if j['method'] == method} == set(LR_GRIDS[method]) - {.002}
    assert tuple(j['lr'] for j in first if j['method'] == 'prefix_standard') == LR_GRIDS['prefix_standard']
    unique = {(j['method'], j['lr'], j['max_grad_norm']) for j in first + second}
    assert len(unique) == 23
    finals = final_trials(configs())
    assert len(finals) == 18 and Counter(j['seed'] for j in finals) == {s: 6 for s in FINAL_SEEDS}


def test_selection_uses_final_top1_ties_smaller_clip_lr():
    first = [(j, Path(j['name']), {'final': {'test_top1': .5}}) for j in first_search_trials()]
    second = [(j, Path(j['name']), {'final': {'test_top1': .5}})
              for j in second_search_trials({'iid_scale': 50., 'momentum_scale': 50.})]
    selected = sweep.select_configs(first, second)
    assert selected['iid_scale'] == dict(lr=.0005, max_grad_norm=50., eps_scale=.1)
    assert selected['momentum_scale'] == dict(lr=.001, max_grad_norm=50., eps_scale=.1, num_bands=4)
    assert selected['prefix_standard']['lr'] == .0005
    assert all(selected[k] == v for k, v in ANCHORS.items())
    second[-1][2]['final']['test_top1'] = .6
    assert sweep.select_configs(first, second)['momentum_scale']['lr'] == .007


@pytest.mark.parametrize('noise,original_method', [('iid', 'dp_adam'),
    ('prefix_bandinvmf', 'dp_adam_bandinvmf_scale'),
    ('momentum_bandinvmf', 'dp_adam_bandinvmf_momentum')])
def test_factorization_matches_exp1e_and_geometry_invariant(noise, original_method):
    from exp1e.bandinvmf import build_matrices as original
    d, c, w = build_matrices(noise, 250, 4, .9)
    for actual, expected in zip((d, c, w), original(original_method, 250, 4, .9)):
        np.testing.assert_array_equal(actual, expected)
    row = [method for method, cell in CELLS.items() if cell['noise'] == noise]
    assert len(row) == 2
    for method in row:
        for actual, expected in zip(build_matrices(CELLS[method]['noise'], 250, 4, .9), (d, c, w)):
            np.testing.assert_array_equal(actual, expected)
    D = materialize(d, 250)
    np.testing.assert_allclose(D @ c, np.eye(250), atol=2e-14)
    if noise == 'iid':
        np.testing.assert_array_equal(d, [1.])
    elif noise == 'prefix_bandinvmf':
        np.testing.assert_array_equal(materialize(w, 250), np.tril(np.ones((250, 250))))
    else:
        np.testing.assert_allclose(w, np.cumsum(.9 ** np.arange(250)), atol=2e-14)


@pytest.mark.parametrize('noise', ['iid', 'prefix_bandinvmf', 'momentum_bandinvmf'])
def test_privacy_recalibrates_for_every_clip(noise):
    _, c, _ = build_matrices(noise, 250, 4, .9)
    deviations = []
    for clip in (1, *CLIPS):
        cfg = copy.deepcopy(CONFIG)
        cfg['privacy']['max_grad_norm'] = clip
        result = calibrate(c, cfg)
        assert result['innovation_std_sum'] == pytest.approx(clip * result['sensitivity'] / result['target_mu'])
        assert gdp_delta(result['target_mu'], 8.) == pytest.approx(1e-5)
        assert result['sampling_amplification'] is False
        deviations.append(result['innovation_std_sum'])
    np.testing.assert_allclose(np.array(deviations) / deviations[0], [1, *CLIPS])


def test_dataset_local_download_false(monkeypatch):
    from exp2.train import datasets
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **kw: pytest.fail('network requested'))
    _, cfg = models.pretrained_vit()
    train_transform, test_transform = image_transforms(cfg)
    train = datasets.CIFAR100(ROOT / 'data', train=True, transform=train_transform, download=False)
    test = datasets.CIFAR100(ROOT / 'data', train=False, transform=test_transform, download=False)
    assert len(train) == 50000 and len(test) == 10000
    assert train[0][0].shape == test[0][0].shape == (3, 224, 224)
    # Also inspect the actual training entrypoint, so the test covers its call sites.
    import inspect
    from exp2.train import run
    source = inspect.getsource(run)
    assert source.count('download=False)') == 2
    assert "ROOT / config['data_root']" in source


@pytest.mark.parametrize('seed', [SEARCH_SEED, *FINAL_SEEDS])
def test_local_pretrained_full_finetuning_paired_rng(seed, monkeypatch):
    from PIL import Image
    import timm
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **kw: pytest.fail('network requested'))
    path = models.checkpoint_path()
    assert path.resolve().is_relative_to(ROOT / 'cache')
    seed_all(seed)
    reference = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    reference.load_state_dict(load_file(str(path)), strict=True)
    expected = models.ViTTiny()
    expected.load_backbone(reference)
    digest = models.initialization_digest(expected)
    expected_rng = torch.get_rng_state().clone()
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    cfg = reference.pretrained_cfg
    augmentation = image_transforms(cfg)[0](image)
    for method in CELLS:
        seed_all(seed)
        model, metadata = models.pretrained_vit()
        assert models.initialization_digest(model) == digest
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        torch.testing.assert_close(image_transforms(metadata)[0](image), augmentation, rtol=0, atol=0)
        assert metadata['checkpoint_sha256'] == models.checkpoint_sha256(path)
        assert len(metadata['checkpoint_sha256']) == 64
        assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
        assert model.head.out_features == 100
        assert sum(p.numel() for p in model.parameters()) > 5_000_000
        np.testing.assert_array_equal(coordinate_indices(sum(p.numel() for p in model.parameters())),
                                      coordinate_indices(sum(p.numel() for p in expected.parameters())))


def test_missing_checkpoint_fails_without_download(tmp_path, monkeypatch):
    monkeypatch.setattr(models, 'CHECKPOINT_CACHE', tmp_path)
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **kw: pytest.fail('network requested'))
    with pytest.raises(FileNotFoundError):
        models.pretrained_vit()


@pytest.mark.parametrize('scaled', [False, True])
def test_exact_clipping_matches_per_example_reference(scaled):
    torch.manual_seed(42)
    base = tiny()
    wrapped = (ScaledGhostModule(copy.deepcopy(base), 1.) if scaled else
               GradSampleModuleFastGradientClipping(copy.deepcopy(base), loss_reduction='sum', max_grad_norm=1.))
    opt = torch.optim.Adam(wrapped.parameters())
    for p in wrapped.parameters():
        opt.state[p].update(step=torch.tensor(3.), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.rand_like(p) * .001)
    logical = LogicalBatch(wrapped, opt, 4, 1000, noise=None, scaled=scaled, eps_scale=.1)
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
            total.add_(p.grad, alpha=min(1., 1 / float(norm)))
    _, clipped = clipped_microbatch(wrapped, x, y, 1.)
    assert clipped == sum(n > 1 for n in norms)
    for p, total in zip(wrapped.parameters(), sums):
        torch.testing.assert_close(p.grad, total, rtol=5e-5, atol=1e-6)


@pytest.mark.parametrize('method', list(CELLS))
def test_four_microbatches_once_noise_once_adam_previous_vhat_and_current_r(method):
    model = tiny()
    opt = torch.optim.Adam(model.parameters(), eps=1e-8, betas=(.9, .999))
    d, _, _ = build_matrices(CELLS[method]['noise'], 3, 4, .9)
    noise = BandInvMFNoise(model.parameters(), d, .1, 3, 10)
    scaled = CELLS[method]['geometry'] == 'scale'
    logical = LogicalBatch(model, opt, 4, 1000, noise, scaled=scaled, eps_scale=.1)
    trace = MechanismTrace(model.parameters(), scaled)
    for step in range(3):
        expected = [torch.zeros_like(p) if step == 0 else opt.state[p]['exp_avg_sq'] / (1 - .999 ** step)
                    for p in model.parameters()]
        for micro in range(4):
            logical.begin_microbatch()
            if scaled:
                for p, vhat in zip(model.parameters(), expected):
                    torch.testing.assert_close(p._logical_scale, 1 / (vhat.sqrt() + .1))
            for p in model.parameters():
                p.grad = torch.ones_like(p) * (step + 1)
            completed = logical.finish_microbatch()
            assert completed == (micro == 3)
            assert noise.step_count == logical.optimizer_steps == step + int(completed)
            if completed:
                trace.record(opt)
        actual_r, actual_s = [], []
        for p, prev in zip(model.parameters(), expected):
            vhat = opt.state[p]['exp_avg_sq'] / (1 - .999 ** (step + 1))
            scale = 1 / (prev.sqrt() + .1) if scaled else torch.ones_like(p)
            actual_r.append((1 / (vhat.sqrt() + 1e-8) / scale).flatten().detach().numpy())
            actual_s.append(scale.flatten().detach().numpy())
        np.testing.assert_allclose(trace.r_trace[-1], np.concatenate(actual_r)[trace.indices], rtol=1e-6)
        np.testing.assert_allclose(trace.s_trace[-1], np.concatenate(actual_s)[trace.indices], rtol=1e-6)
    assert all(int(opt.state[p]['step']) == 3 for p in model.parameters())
    assert len(trace.r_trace) == 3 and len(trace.indices) == 2048


@pytest.mark.parametrize('noise_kind', ['prefix_bandinvmf', 'momentum_bandinvmf'])
def test_noise_transform_is_actual_innovation_convolution(noise_kind):
    d, _, _ = build_matrices(noise_kind, 8, 4, .9)
    p = torch.nn.Parameter(torch.zeros(2, 3))
    noise = BandInvMFNoise([p], d, 2.3, 8, 71)
    generator = torch.Generator().manual_seed(71)
    innovations = np.stack([(torch.randn(p.shape, generator=generator) * 2.3).numpy().ravel() for _ in range(8)])
    outputs = np.stack([noise.next()[0].numpy().ravel() for _ in range(8)])
    np.testing.assert_allclose(outputs, materialize(d, 8) @ innovations, rtol=1e-6, atol=1e-6)


def test_coordinate_indices_and_mechanism_metrics():
    indices = coordinate_indices(5_543_716)
    assert len(indices) == len(set(indices)) == 2048
    np.testing.assert_array_equal(indices, coordinate_indices(5_543_716))
    r = np.arange(1, 2049.)
    stats = mechanism_metrics(r, r / 2)
    assert stats['temporal_drift'] == pytest.approx(np.log(2))
    assert stats['r_cv'] == pytest.approx(r.std() / r.mean())
    assert stats['r_anisotropy'] == pytest.approx(np.quantile(r, .9) / np.quantile(r, .1))
    assert mechanism_metrics(r)['temporal_drift'] is None


@pytest.mark.parametrize('noise_kind', ['prefix_bandinvmf', 'momentum_bandinvmf'])
def test_mf_distortion_matches_explicit_frobenius_actual_workload(noise_kind):
    d, _, w = build_matrices(noise_kind, 8, 4, .9)
    M, W = materialize(d, 8) * 1.37, materialize(w, 8)
    r = np.exp(np.random.default_rng(1).normal(size=(8, 9)))
    result = mf_distortion(r, M, W)
    ratios = [np.linalg.norm(W @ np.diag(r[:, j]) @ M, 'fro') /
              (np.sqrt(np.mean(r[:, j] ** 2)) * np.linalg.norm(W @ M, 'fro')) for j in range(9)]
    assert result['ratio_p50'] == pytest.approx(np.median(ratios), rel=1e-12)
    assert result['ratio_p10'] == pytest.approx(np.quantile(ratios, .1), rel=1e-12)
    assert result['ratio_p90'] == pytest.approx(np.quantile(ratios, .9), rel=1e-12)
    assert result['logabs_mean'] == pytest.approx(np.abs(np.log(ratios)).mean(), rel=1e-12)
    assert result['logabs_median'] == pytest.approx(np.median(np.abs(np.log(ratios))), rel=1e-12)
    constant = mf_distortion(np.ones((8, 9)) * 2, M, W)
    assert constant['ratio_p50'] == pytest.approx(1.)
    assert constant['logabs_mean'] == pytest.approx(0., abs=1e-14)


def test_factorial_formulas():
    accuracy = dict(iid_standard=.4, iid_scale=.45, prefix_standard=.42, prefix_scale=.52,
                    momentum_standard=.50, momentum_scale=.53)
    assert factorial_effects(accuracy) == pytest.approx(dict(G_iid=.05, G_prefix=.10, G_momentum=.03,
                                                          I_prefix=.05, I_momentum=-.02))


def test_launcher_four_gpus_refills_failures_and_completed_reuse(tmp_path, monkeypatch):
    active, peak, launched, finished = {}, [0], [], []
    jobs = [dict(j, name=f'trial_{i}') for i, j in enumerate(final_trials(configs())[:9])]
    class Process:
        def __init__(self, cmd, cwd, env, stdout, stderr):
            gpu = int(env['CUDA_VISIBLE_DEVICES'])
            assert gpu in (0, 1, 2, 3) and gpu not in active
            assert stderr == sweep.subprocess.STDOUT and stdout.name.endswith('train.log')
            assert cmd[cmd.index('-m') + 1] == 'exp2.train' and '--smoke' not in cmd
            self.gpu, self.polls = gpu, 0
            active[gpu] = self
            peak[0] = max(peak[0], len(active))
            launched.append((gpu, len(finished)))
            self.fail = len(launched) == 2
        def poll(self):
            self.polls += 1
            if self.polls < (2 if self.gpu == 1 else 4):
                return None
            del active[self.gpu]
            finished.append(self.gpu)
            return 7 if self.fail else 0
    monkeypatch.setattr(sweep.subprocess, 'Popen', Process)
    monkeypatch.setattr(sweep.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sweep, 'read_completed', lambda *a: {'status': 'completed'})
    with pytest.raises(RuntimeError, match='Trial failed'):
        sweep.run_queue(jobs, tmp_path, ROOT / 'exp2/config.yaml')
    assert peak[0] == 4 and len(launched) == 9 and len(finished) == 9
    assert launched[4] == (1, 1)  # GPU 1 gets a new trial before the other GPUs finish.
    assert not active
    def forbidden(*a, **kw):
        pytest.fail('completed trial relaunched')
    monkeypatch.setattr(sweep.subprocess, 'Popen', forbidden)
    assert len(sweep.run_queue(jobs, tmp_path, ROOT / 'exp2/config.yaml')) == 9


def test_search_selection_and_final_barriers(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, 'ROOT', tmp_path)
    monkeypatch.setattr(sweep.torch.cuda, 'device_count', lambda: 4)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    phases = []
    def queue(jobs, base, config_path, smoke=False):
        phases.append(len(jobs))
        if len(phases) == 1:
            assert len(jobs) == 16
        elif len(phases) == 2:
            assert len(jobs) == 7 and not (tmp_path / 'exp2/results/selected_configs.json').exists()
        else:
            assert phases == [16, 7, 18]
            assert (tmp_path / 'exp2/results/selected_configs.json').is_file()
        return [(j, base / j['name'], {'final': {'test_top1': .5}, 'initialization_sha256': 'same'}) for j in jobs]
    monkeypatch.setattr(sweep, 'run_queue', queue)
    monkeypatch.setattr(sweep, 'aggregate_final', lambda *a: None)
    sweep.main(SimpleNamespace(config=ROOT / 'exp2/config.yaml', smoke=False))
    assert phases == [16, 7, 18]
    assert len(list(csv.DictReader((tmp_path / 'exp2/results/search_summary.csv').open()))) == 23
    phases.clear()
    def failure(*a, **kw):
        phases.append('failed')
        raise RuntimeError('trial failure')
    monkeypatch.setattr(sweep, 'run_queue', failure)
    with pytest.raises(RuntimeError):
        sweep.main(SimpleNamespace(config=ROOT / 'exp2/config.yaml', smoke=False))
    assert phases == ['failed']


def test_final_aggregation_actual_trial_matrices_and_paired_statistics(tmp_path):
    from exp2.diagnostics import write_csv
    results = []
    chosen = configs()
    for job in final_trials(chosen):
        directory = tmp_path / job['name']
        directory.mkdir(parents=True)
        method, seed = job['method'], job['seed']
        cell = CELLS[method]
        d, c, w = build_matrices(cell['noise'], 8, 4, .9)
        sigma = 1. + job['max_grad_norm']
        W, M = materialize(w, 8), materialize(d, 8) * sigma
        np.savez(directory / 'matrices.npz', noising_coefficients=d, strategy=c,
                 workload_coefficients=w, W=W, M=M, innovation_std_sum=sigma)
        # Deliberately different from another workload to catch rebuilding W.
        r = np.exp(np.arange(8)[:, None] * (.02 if cell['geometry'] == 'scale' else .2))
        r = np.repeat(r, 2048, axis=1)
        np.savez(directory / 'mechanism_trace.npz', coordinate_indices=np.arange(2048),
                 r_trace=r, s_trace=np.ones_like(r))
        write_csv(directory / 'mechanism_metrics.csv', [dict(step=i + 1, **mechanism_metrics(row, r[i-1] if i else None))
                                                         for i, row in enumerate(r)])
        np.save(directory / 'train_order.npy', np.arange(50000))
        (directory / 'config.yaml').write_text(yaml.safe_dump(dict(num_workers=2, augmentation_rng_convention='paired')))
        idx = list(CELLS).index(method)
        accuracy = .3 + .02 * idx + .01 * (seed - FINAL_SEEDS[0]) * (idx + 1)
        summary = dict(initialization_sha256=f'init_{seed}', classifier_initialization_sha256=f'head_{seed}',
                       pretrained_checkpoint_sha256='checkpoint', augmentation_first_batch_sha256=[f'aug_{seed}'],
                       final=dict(test_top1=accuracy))
        results.append((job, directory, summary))
    sweep.aggregate_final(results, chosen, tmp_path)
    final = json.loads((tmp_path / 'final_summary.json').read_text())
    assert final['iid_standard']['top1_mean'] == pytest.approx(.31)
    assert final['iid_standard']['top1_std'] == pytest.approx(.01)
    effects = json.loads((tmp_path / 'factorial_effects.json').read_text())
    assert len(effects['paired']) == 3
    assert effects['aggregate']['G_iid']['mean'] == pytest.approx(.03)
    assert effects['aggregate']['G_iid']['sample_std'] == pytest.approx(.01)
    assert effects['aggregate']['I_prefix']['mean'] == pytest.approx(0., abs=1e-15)
    assert len(list(csv.DictReader((tmp_path / 'final_multiseed.csv').open()))) == 18
    distortions = list(csv.DictReader((tmp_path / 'mf_distortion.csv').open()))
    assert len(distortions) == 12
    for job, directory, _ in results:
        if CELLS[job['method']]['noise'] == 'iid':
            continue
        with np.load(directory / 'mechanism_trace.npz') as trace, np.load(directory / 'matrices.npz') as matrices:
            expected = mf_distortion(trace['r_trace'], matrices['M'], matrices['W'])
        row = next(r for r in distortions if r['method'] == job['method'] and int(r['seed']) == job['seed'])
        assert float(row['ratio_p50']) == pytest.approx(expected['ratio_p50'])
        assert float(row['logabs_mean']) == pytest.approx(expected['logabs_mean'])
    mechanisms = json.loads((tmp_path / 'mechanism_summary.json').read_text())
    assert len(mechanisms['pairs']) == 3 and len(mechanisms['cells']) == 6
    for method, cell in mechanisms['cells'].items():
        assert all(key in cell for key in ('top1_mean', 'top1_std', 'r_cv_mean', 'r_anisotropy_mean', 'temporal_drift_mean'))
        assert cell['temporal_drift_mean'] == pytest.approx(.02 if CELLS[method]['geometry'] == 'scale' else .2)
        assert ('mf_distortion_logabs_mean' in cell) == (CELLS[method]['noise'] != 'iid')
    # Saved artifacts must fail verification if pairing breaks.
    results[1][2]['initialization_sha256'] = 'wrong'
    with pytest.raises(AssertionError):
        sweep.verify_pairing(results)


def test_completed_validation_rejects_incomplete_and_conflicting_trial(tmp_path):
    job = final_trials(configs())[0]
    with pytest.raises(FileNotFoundError):
        sweep.read_completed(tmp_path, job, ROOT / 'exp2/config.yaml')
    summary = dict(status='completed', smoke=False, method=job['method'], seed=job['seed'],
                   lr=job['lr'] * 2, max_grad_norm=job['max_grad_norm'], eps_scale=None, num_bands=None)
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(AssertionError):
        sweep.read_completed(tmp_path, job, ROOT / 'exp2/config.yaml')
