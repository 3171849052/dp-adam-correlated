import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from jax_privacy.matrix_factorization import sensitivity as jax_sensitivity
import jax.numpy as jnp

from exp1.bandinvmf import BandInvMFNoise, build_matrices
from exp1.model import ViTTiny
from exp1.privacy import (calibrate, epsilon_from_mu, fixed_epoch_sensitivity,
                          gdp_delta, target_mu)
from exp1.scale import (LogicalBatch, ScaledGhostModule, clipped_microbatch,
                        freeze_adam_scale)
from exp1.train import load_config

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]


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
    coefficients, strategy, _ = build_matrices(method, 250, 1, 0.9)
    np.testing.assert_array_equal(coefficients, [1])
    np.testing.assert_array_equal(strategy, np.eye(250))
    p = nn.Parameter(torch.zeros(8))
    noise = BandInvMFNoise([p], coefficients, 2.3, 250, seed=5)
    gen = torch.Generator().manual_seed(5)
    for _ in range(5):
        torch.testing.assert_close(noise.next()[0], torch.randn(8, generator=gen) * 2.3)
    assert noise.history == []


def test_correlated_noise_is_toeplitz_convolution():
    coefficients, _, _ = build_matrices('dp_adam_bandinvmf_momentum', 250, 4, 0.9)
    p = nn.Parameter(torch.zeros(3))
    noise = BandInvMFNoise([p], coefficients, 0.7, 250, seed=91)
    generator = torch.Generator().manual_seed(91)
    zs = [torch.randn(3, generator=generator) * 0.7 for _ in range(9)]
    for t in range(9):
        expected = sum(float(coefficients[lag]) * zs[t-lag] for lag in range(min(t+1, 4)))
        torch.testing.assert_close(noise.next()[0], expected)


def test_workload_only_uses_beta1():
    _, _, workload = build_matrices('dp_adam_bandinvmf_momentum', 250, 4, 0.9)
    np.testing.assert_allclose(workload, np.cumsum(0.9 ** np.arange(250)), rtol=1e-12)


@pytest.mark.parametrize('method', ['dp_adam', 'dp_adam_bandinvmf_momentum', 'dp_adam_bandinvmf_scale'])
def test_fixed_epoch_sensitivity_and_shared_gdp(method):
    config = load_config(ROOT / 'exp1/config.yaml')
    _, strategy, _ = build_matrices(method, 250, 4, 0.9)
    calibration = calibrate(strategy, config)
    assert (calibration['k'], calibration['b_participation']) == (5, 50)
    assert not calibration['sampling_amplification']
    expected = jax_sensitivity.fixed_epoch_sensitivity(jnp.asarray(strategy), epochs=5)
    assert calibration['sensitivity'] == pytest.approx(expected)
    assert calibration['sensitivity'] / calibration['innovation_std_sum'] == pytest.approx(target_mu(8, 1e-5))
    if method == 'dp_adam':
        assert expected == pytest.approx(np.sqrt(5))


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
    base = ViTTiny(embed_dim=12, depth=1, heads=3, mlp_ratio=2)
    model = ScaledGhostModule(copy.deepcopy(base), max_grad_norm=1)
    scales = [torch.rand_like(p) + 0.1 for p in model.parameters()]
    for p, s in zip(model.parameters(), scales):
        p._logical_scale = s
    x, y = torch.randn(2, 3, 32, 32), torch.tensor([0, 3])
    expected = explicit_norms(base, x, y, scales)
    clipped_microbatch(model, x, y, 1)
    torch.testing.assert_close(model.get_norm_sample(), expected, rtol=3e-5, atol=1e-5)


def test_vit_unit_scale_matches_ghost_sequence_norms():
    base = ViTTiny(embed_dim=12, depth=1, heads=3, mlp_ratio=2)
    scaled = ScaledGhostModule(copy.deepcopy(base), max_grad_norm=1)
    ghost = GradSampleModuleFastGradientClipping(copy.deepcopy(base), loss_reduction='sum')
    x, y = torch.randn(2, 3, 32, 32), torch.tensor([0, 3])
    for model in (scaled, ghost):
        nn.functional.cross_entropy(model(x), y, reduction='sum').backward()
    torch.testing.assert_close(scaled.get_norm_sample(), ghost.get_norm_sample(), rtol=3e-5, atol=1e-5)


def test_20_microbatches_one_optimizer_and_noise_step_and_frozen_scale():
    model = ScaledGhostModule(mlp(), max_grad_norm=1)
    opt = torch.optim.Adam(model.parameters())
    noise = BandInvMFNoise(model.parameters(), [1, -0.5, -0.125, -0.0625], 1, 250, 3)
    logical = LogicalBatch(model, opt, 20, 1000, noise, scaled=True)
    frozen = None
    for micro in range(20):
        logical.begin_microbatch()
        if frozen is None:
            frozen = [p._logical_scale.clone() for p in model.parameters()]
        for p, s in zip(model.parameters(), frozen):
            torch.testing.assert_close(p._logical_scale, s, rtol=0, atol=0)
            p.grad = torch.ones_like(p)
        advanced = logical.finish_microbatch()
        assert advanced == (micro == 19)
        assert noise.step_count == logical.optimizer_steps == (1 if micro == 19 else 0)
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


def test_exactly_250_band_steps_then_error():
    model = nn.Linear(1, 1, bias=False)
    noise = BandInvMFNoise(model.parameters(), [1, -0.5, -0.125, -0.0625], 1, 250, 3)
    opt = torch.optim.Adam(model.parameters())
    logical = LogicalBatch(model, opt, 20, 1000, noise)
    for _ in range(250 * 20):
        logical.begin_microbatch()
        model.weight.grad = torch.ones_like(model.weight)
        logical.finish_microbatch()
    assert noise.step_count == logical.optimizer_steps == 250
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
    _, strategy, _ = build_matrices('dp_adam_bandinvmf_momentum', 250, 4, 0.9)
    full = fixed_epoch_sensitivity(strategy, 5, 50)
    prefix = [fixed_epoch_sensitivity(strategy, 5, 50, steps=n) for n in (50, 100, 150, 200, 250)]
    assert prefix == sorted(prefix)
    assert prefix[-1] == full
