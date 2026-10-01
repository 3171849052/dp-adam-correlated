import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from jax_privacy.matrix_factorization import sensitivity as jax_sensitivity
import jax.numpy as jnp

from exp1.bandinvmf import BandInvMFNoise, build_matrices
from exp1.model import ViTTiny, pretrained_vit, initialization_digest
from exp1.privacy import (calibrate, epsilon_from_mu, fixed_epoch_sensitivity,
                          gdp_delta, target_mu)
from exp1.scale import (LogicalBatch, ScaledGhostModule, clipped_microbatch,
                        freeze_adam_scale)
from exp1.train import load_config

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(ROOT / 'exp1/config.yaml')
T = CONFIG['total_steps']
K = CONFIG['privacy']['k']
B = CONFIG['privacy']['b_participation']


@pytest.mark.parametrize('physical,accumulation', [(50, 20), (250, 4), (1000, 1)])
def test_config_accepts_microbatch_partitions(tmp_path, physical, accumulation):
    config = yaml.safe_load((ROOT / 'exp1/config.yaml').read_text())
    config.update(physical_batch_size=physical, gradient_accumulation=accumulation)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    resolved = load_config(path)
    assert resolved['total_steps'] == resolved['epochs'] * resolved['privacy']['b_participation']
    assert resolved['physical_batch_size'] == physical


@pytest.mark.parametrize('physical,accumulation', [(250, 20), (0, 4), (250, -4),
                                                 (250.0, 4), (250, True)])
def test_config_rejects_invalid_microbatch_partitions(tmp_path, physical, accumulation):
    config = yaml.safe_load((ROOT / 'exp1/config.yaml').read_text())
    config.update(physical_batch_size=physical, gradient_accumulation=accumulation)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='physical_batch_size|gradient_accumulation'):
        load_config(path)


def mlp():
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(4, 7), nn.LayerNorm(7), nn.GELU(), nn.Linear(7, 3))


def explicit_norms(model, x, y, scales):
    norms = []
    for xi, yi in zip(x, y):
        model.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(xi.unsqueeze(0)), yi.unsqueeze(0)).backward()
        norms.append(torch.stack([(p.grad * s).square().sum()
                                 for p, s in zip(model.parameters(), scales)]).sum().sqrt())
    return torch.stack(norms)


def test_gdp_calibration():
    mu = target_mu(8, 1e-5)
    assert gdp_delta(mu, 8) == pytest.approx(1e-5)
    assert epsilon_from_mu(mu, 1e-5) == pytest.approx(8)


@pytest.mark.parametrize('method', ['dp_adam_bandinvmf_momentum', 'dp_adam_bandinvmf_scale'])
def test_one_band_is_iid(method):
    coefficients, strategy, _ = build_matrices(method, T, 1, 0.9)
    np.testing.assert_array_equal(coefficients, [1])
    np.testing.assert_array_equal(strategy, np.eye(T))
    p = nn.Parameter(torch.zeros(8))
    noise = BandInvMFNoise([p], coefficients, 2.3, T, seed=5)
    gen = torch.Generator().manual_seed(5)
    for _ in range(5):
        torch.testing.assert_close(noise.next()[0], torch.randn(8, generator=gen) * 2.3)
    assert noise.history == []


def test_correlated_noise_is_toeplitz_convolution():
    coefficients, _, _ = build_matrices('dp_adam_bandinvmf_momentum', T, 4, 0.9)
    p = nn.Parameter(torch.zeros(3))
    noise = BandInvMFNoise([p], coefficients, 0.7, T, seed=91)
    generator = torch.Generator().manual_seed(91)
    zs = [torch.randn(3, generator=generator) * 0.7 for _ in range(9)]
    for t in range(9):
        expected = sum(float(coefficients[lag]) * zs[t-lag] for lag in range(min(t+1, 4)))
        torch.testing.assert_close(noise.next()[0], expected)


def test_workload_only_uses_beta1():
    _, _, workload = build_matrices('dp_adam_bandinvmf_momentum', T, 4, 0.9)
    np.testing.assert_allclose(workload, np.cumsum(0.9 ** np.arange(T)), rtol=1e-12)


@pytest.mark.parametrize('method', ['dp_adam', 'dp_adam_bandinvmf_momentum', 'dp_adam_bandinvmf_scale'])
def test_fixed_epoch_sensitivity_and_shared_gdp(method):
    config = load_config(ROOT / 'exp1/config.yaml')
    _, strategy, _ = build_matrices(method, T, 4, 0.9)
    calibration = calibrate(strategy, config)
    assert (calibration['k'], calibration['b_participation']) == (K, B)
    assert not calibration['sampling_amplification']
    expected = jax_sensitivity.fixed_epoch_sensitivity(jnp.asarray(strategy), epochs=K)
    assert calibration['sensitivity'] == pytest.approx(expected)
    assert calibration['sensitivity'] / calibration['innovation_std_sum'] == pytest.approx(target_mu(8, 1e-5))
    if method == 'dp_adam':
        assert expected == pytest.approx(np.sqrt(K))


def test_scaled_unit_norm_equals_opacus_ghost_norm():
    base = mlp()
    scaled = ScaledGhostModule(copy.deepcopy(base), max_grad_norm=1)
    ghost = GradSampleModuleFastGradientClipping(copy.deepcopy(base), loss_reduction='sum')
    x, y = torch.randn(6, 4), torch.tensor([0, 1, 2, 0, 1, 2])
    for model in (scaled, ghost):
        nn.functional.cross_entropy(model(x), y, reduction='sum').backward()
    torch.testing.assert_close(scaled.get_norm_sample(), ghost.get_norm_sample(), rtol=2e-5, atol=1e-6)


def test_arbitrary_scaled_norm_and_clipped_sum_are_exact():
    base = mlp()
    model = ScaledGhostModule(copy.deepcopy(base), max_grad_norm=0.6)
    scales = [torch.rand_like(p) * 4 + 0.1 for p in model.parameters()]
    for p, s in zip(model.parameters(), scales):
        p._logical_scale = s
    x, y = torch.randn(6, 4), torch.tensor([0, 1, 2, 0, 1, 2])
    expected_norms = explicit_norms(base, x, y, scales)
    clipped_microbatch(model, x, y, max_grad_norm=0.6)
    torch.testing.assert_close(model.get_norm_sample(), expected_norms, rtol=2e-5, atol=1e-6)
    expected = [torch.zeros_like(p) for p in base.parameters()]
    for i in range(6):
        base.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(base(x[i:i+1]), y[i:i+1]).backward()
        coef = min(1, 0.6 / expected_norms[i].item())
        for total, p in zip(expected, base.parameters()):
            total.add_(p.grad, alpha=coef)
    for p, total in zip(model.parameters(), expected):
        torch.testing.assert_close(p.grad, total, rtol=2e-5, atol=1e-6)


def test_vit_scaled_norm_includes_tokens_and_sequence_linear():
    base = ViTTiny(patch_size=4, image_size=32, embed_dim=12, depth=1, heads=3, mlp_ratio=2)
    model = ScaledGhostModule(copy.deepcopy(base), max_grad_norm=1)
    scales = [torch.rand_like(p) + 0.1 for p in model.parameters()]
    for p, s in zip(model.parameters(), scales):
        p._logical_scale = s
    x, y = torch.randn(2, 3, 32, 32), torch.tensor([0, 3])
    expected = explicit_norms(base, x, y, scales)
    clipped_microbatch(model, x, y, 1)
    torch.testing.assert_close(model.get_norm_sample(), expected, rtol=3e-5, atol=1e-5)


def test_vit_unit_scale_matches_ghost_sequence_norms():
    base = ViTTiny(patch_size=4, image_size=32, embed_dim=12, depth=1, heads=3, mlp_ratio=2)
    scaled = ScaledGhostModule(copy.deepcopy(base), max_grad_norm=1)
    ghost = GradSampleModuleFastGradientClipping(copy.deepcopy(base), loss_reduction='sum')
    x, y = torch.randn(2, 3, 32, 32), torch.tensor([0, 3])
    for model in (scaled, ghost):
        nn.functional.cross_entropy(model(x), y, reduction='sum').backward()
    torch.testing.assert_close(scaled.get_norm_sample(), ghost.get_norm_sample(), rtol=3e-5, atol=1e-5)


@pytest.mark.parametrize('accumulation', [20, 4])
def test_accumulation_one_optimizer_and_noise_step_and_frozen_scale(accumulation):
    model = ScaledGhostModule(mlp(), max_grad_norm=1)
    opt = torch.optim.Adam(model.parameters())
    noise = BandInvMFNoise(model.parameters(), [1, -0.5, -0.125, -0.0625], 1, T, 3)
    logical = LogicalBatch(model, opt, accumulation, 1000, noise, scaled=True)
    frozen = None
    for micro in range(accumulation):
        logical.begin_microbatch()
        if frozen is None:
            frozen = [p._logical_scale.clone() for p in model.parameters()]
        for p, s in zip(model.parameters(), frozen):
            torch.testing.assert_close(p._logical_scale, s, rtol=0, atol=0)
            p.grad = torch.ones_like(p)
        advanced = logical.finish_microbatch()
        assert advanced == (micro == accumulation - 1)
        assert noise.step_count == logical.optimizer_steps == (1 if micro == accumulation - 1 else 0)
    assert all(int(opt.state[p]['step'].item()) == 1 for p in model.parameters())
    logical.begin_microbatch()
    assert any(not torch.equal(p._logical_scale, s) for p, s in zip(model.parameters(), frozen))


def test_scale_uses_previous_bias_corrected_second_moment():
    model = mlp()
    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999))
    for p in model.parameters():
        p.grad = torch.full_like(p, 2)
    opt.step()
    for scale in freeze_adam_scale(opt, 0.01):
        torch.testing.assert_close(scale, torch.full_like(scale, 1 / 2.01))


def test_full_derived_schedule_then_error():
    model = nn.Linear(1, 1, bias=False)
    noise = BandInvMFNoise(model.parameters(), [1, -0.5, -0.125, -0.0625], 1, T, 3)
    opt = torch.optim.Adam(model.parameters())
    logical = LogicalBatch(model, opt, 20, 1000, noise)
    for _ in range(T * 20):
        logical.begin_microbatch()
        model.weight.grad = torch.ones_like(model.weight)
        logical.finish_microbatch()
    assert noise.step_count == logical.optimizer_steps == T
    with pytest.raises(AssertionError, match='exhausted'):
        noise.next()


def test_no_noise_unbounded_scale_matches_adam_over_multiple_steps():
    base = mlp()
    model = ScaledGhostModule(copy.deepcopy(base), max_grad_norm=float('inf'))
    opt_plain = torch.optim.Adam(base.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    opt_scale = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    logical = LogicalBatch(model, opt_scale, 20, 40, scaled=True)
    for _ in range(3):
        x, y = torch.randn(40, 4), torch.randint(3, (40,))
        opt_plain.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(base(x), y).backward()
        opt_plain.step()
        for micro in range(20):
            logical.begin_microbatch()
            clipped_microbatch(model, x[2*micro:2*micro+2], y[2*micro:2*micro+2], float('inf'))
            logical.finish_microbatch()
        for a, b in zip(base.parameters(), model.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6)


def test_prefix_accounting_reaches_full_target():
    _, strategy, _ = build_matrices('dp_adam_bandinvmf_momentum', T, 4, 0.9)
    full = fixed_epoch_sensitivity(strategy, K, B)
    prefix = [fixed_epoch_sensitivity(strategy, K, B, steps=n) for n in range(B, T + 1, B)]
    assert prefix == sorted(prefix)
    assert prefix[-1] == full


def test_derived_schedule_and_bands(tmp_path):
    assert (K, B, T) == (5, 50, 250)
    assert CONFIG['bandinvmf']['num_bands'] == 4
    raw = yaml.safe_load((ROOT / 'exp1/config.yaml').read_text())
    assert 'total_steps' not in raw
    assert 'k' not in raw['privacy'] and 'b_participation' not in raw['privacy']
    raw.update(epochs=3, dataset_size=120, logical_batch_size=20,
               physical_batch_size=2, gradient_accumulation=10)
    path = tmp_path / 'derived.yaml'
    path.write_text(yaml.safe_dump(raw))
    derived = load_config(path)
    assert (derived['privacy']['k'], derived['privacy']['b_participation'], derived['total_steps']) == (3, 6, 18)
    raw['dataset_size'] = 121
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(AssertionError):
        load_config(path)


def test_scale_prefix_workload():
    _, _, workload = build_matrices('dp_adam_bandinvmf_scale', T, 4, 0.9)
    np.testing.assert_array_equal(workload, np.ones(T))


def test_pretrained_mapping_random_head_and_equivalence():
    import timm
    torch.manual_seed(CONFIG['seed'])
    model, cfg = pretrained_vit()
    reference = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    mapped = ViTTiny()
    mapped.load_backbone(reference)
    for name, value in model.state_dict().items():
        if not name.startswith('head.'):
            torch.testing.assert_close(value, mapped.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(model.patch.weight, reference.patch_embed.proj.weight.flatten(1), rtol=0, atol=0)
    assert not torch.equal(model.patch.weight, ViTTiny().patch.weight)
    assert model.head.out_features == 100
    assert not torch.equal(model.head.weight, reference.head.weight[:100])
    assert not torch.equal(model.head.weight, mapped.head.weight)
    assert all(p.requires_grad for p in model.parameters())
    reference.head = copy.deepcopy(model.head)
    model.eval(); reference.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224)
        torch.testing.assert_close(model(x), reference(x), rtol=2e-4, atol=2e-5)
    torch.manual_seed(CONFIG['seed'])
    repeated, _ = pretrained_vit()
    assert initialization_digest(model) == initialization_digest(repeated)
    assert tuple(cfg['mean']) == (0.5, 0.5, 0.5)


@pytest.mark.parametrize('scaled', [False, True])
def test_full_pretrained_clipping_covers_every_parameter(scaled):
    base, _ = pretrained_vit()
    model = ScaledGhostModule(base, 1) if scaled else GradSampleModuleFastGradientClipping(base, loss_reduction='sum')
    x, y = torch.randn(1, 3, 224, 224), torch.tensor([2])
    clipped_microbatch(model, x, y, 1)
    assert set(model.trainable_parameters) == set(model.parameters())
    assert all(p.requires_grad and p._norm_sample.shape == (1,) and p.grad is not None
               for p in model.parameters())
    assert base.cls.weight._norm_sample.item() > 0
    assert base.position.weight._norm_sample.item() > 0
    explicit = torch.stack([p._norm_sample.square() for p in model.parameters()]).sum(0).sqrt()
    torch.testing.assert_close(explicit, model.get_norm_sample())


def test_sweep_grid_and_summary(tmp_path):
    from collections import Counter
    from exp1.sweep import trials, write_summary
    jobs = trials()
    assert Counter(job['method'] for job in jobs) == dict(adam=3, dp_adam=3,
        dp_adam_bandinvmf_momentum=3, dp_adam_bandinvmf_scale=27)
    assert len({(job['method'], job['name']) for job in jobs}) == 36
    write_summary(jobs, tmp_path / 'sweep', CONFIG, {})
    import csv
    with open(tmp_path / 'sweep_summary.csv') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 36 and all(row['status'] == 'pending' for row in rows)


@pytest.mark.parametrize('exit_code', [0, 1])
def test_launcher_queues_all_jobs_and_reports_failure(tmp_path, monkeypatch, exit_code):
    from argparse import Namespace
    import exp1.sweep as sweep
    launched, live = [], set()

    class Process:
        def __init__(self, cmd, cwd, env, stdout, stderr):
            self.gpu = env['CUDA_VISIBLE_DEVICES']
            assert self.gpu not in live
            live.add(self.gpu)
            assert len(live) <= 4
            launched.append((self.gpu, cmd))
            if exit_code == 0:
                import json
                summary = dict(calibration=dict(target_mu=1.6, sensitivity=2.2, innovation_std_sum=1.4),
                               epochs=[dict(epoch=1, test_top1=0.1)],
                               final=dict(test_top1=0.1, train_loss=4.5, clip_fraction=0.8),
                               wall_seconds=2.0, initialization_sha256='shared')
                directory = Path(cmd[cmd.index('--result-dir') + 1])
                (directory / 'summary.json').write_text(json.dumps(summary))
            self.polls = 0

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                return None
            live.remove(self.gpu)
            return exit_code

    monkeypatch.setattr(sweep, 'ROOT', tmp_path)
    monkeypatch.setattr(sweep.subprocess, 'Popen', Process)
    monkeypatch.setattr(sweep.time, 'sleep', lambda _: None)
    assert sweep.main(Namespace(config=ROOT / 'exp1/config.yaml')) == exit_code
    assert len(launched) == 36 and not live
    assert {gpu for gpu, _ in launched} == {'0', '1', '2', '3'}
    import csv
    with open(tmp_path / 'exp1/results/sweep_summary.csv') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 36 and all(row['status'] == ('failed' if exit_code else 'completed') for row in rows)
    assert all(float(row['wall_seconds']) >= 0 for row in rows)
    if exit_code == 0:
        assert all(float(row['final_test_top1']) == 0.1 for row in rows)


def test_full_pretrained_exact_scaled_norm_against_explicit_gradients():
    base, _ = pretrained_vit()
    model = ScaledGhostModule(copy.deepcopy(base), 1)
    scales = [torch.rand_like(p) * 2 + 0.1 for p in model.parameters()]
    for p, scale in zip(model.parameters(), scales):
        p._logical_scale = scale
    x, y = torch.randn(2, 3, 224, 224), torch.tensor([0, 3])
    expected = explicit_norms(base, x, y, scales)
    clipped_microbatch(model, x, y, 1)
    torch.testing.assert_close(model.get_norm_sample(), expected, rtol=5e-5, atol=1e-5)


def test_pretrained_transforms():
    from exp1.train import image_transforms
    from torchvision import transforms
    from PIL import Image
    train, test = image_transforms(dict(mean=(0.5,)*3, std=(0.5,)*3, crop_pct=0.9))
    assert isinstance(train.transforms[0], transforms.RandomResizedCrop)
    assert isinstance(train.transforms[1], transforms.RandomHorizontalFlip)
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    assert train(image).shape == (3, 224, 224)
    torch.testing.assert_close(test(image), test(image), rtol=0, atol=0)
