import math
import numpy as np
import torch

from exp4.optimizer_uc import UCAdam
from exp4.mechanism import TemporalGaussianNoise, ZeroNoise
from exp4.train import backward_microbatch


class FixedNoise:
    def __init__(self, tensors):
        self.tensors = tensors
        self.step_count = 0

    def next(self):
        self.step_count += 1
        return [t.clone() for t in self.tensors]

    def marginal_std(self, step):
        return 0.7


def test_global_clip_noise_lr_order_and_diagnostics():
    a, b = torch.nn.Parameter(torch.tensor([2., 5.])), torch.nn.Parameter(torch.tensor([-3.]))
    a.grad, b.grad = torch.tensor([3., -4.]), torch.tensor([9.])
    initial = torch.cat([a.detach().clone(), b.detach().clone()])
    noise = FixedNoise([torch.tensor([10., -20.]), torch.tensor([30.])])
    opt = UCAdam([a, b], lr=.2, update_clip_norm=.6, noise=noise)
    stats = opt.step()
    raw = torch.tensor([1., -1., 1.])
    clipped = raw * (.6 / math.sqrt(3))
    expected = initial - .2 * (clipped + torch.tensor([10., -20., 30.]))
    torch.testing.assert_close(torch.cat([a, b]), expected)
    assert np.isclose(stats['raw_update_norm'], math.sqrt(3))
    assert np.isclose(stats['clipped_update_norm'], .6)
    assert np.isclose(stats['clip_scale'], .6 / math.sqrt(3))
    assert stats['clip_indicator'] == 1
    assert np.isclose(stats['parameter_noise_std'], .2 * .7)
    assert np.isclose(stats['signal_parameter_norm'], .2 * .6)
    assert np.isclose(stats['noise_signal_rms_ratio'], .7 / (.6 / math.sqrt(3)))
    assert np.isclose(stats['vhat_min'], 9) and np.isclose(stats['vhat_max'], 81)
    assert np.isclose(stats['vhat_mean'], (9 + 16 + 81) / 3)


def test_hidden_states_exclude_noise_and_clipping_over_multiple_steps():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    q = torch.nn.Parameter(p.detach().clone())
    a = UCAdam([p], lr=.01, update_clip_norm=.1, noise=FixedNoise([torch.tensor([300., -200.])]))
    b = UCAdam([q], lr=.8, update_clip_norm=10., noise=FixedNoise([torch.zeros(2)]))
    expected_m, expected_v = torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    for gradient in (torch.tensor([2., -3.]), torch.tensor([-4., 1.]), torch.tensor([.1, 0.])):
        p.grad, q.grad = gradient.clone(), gradient.clone()
        expected_m = .9 * expected_m + .1 * gradient.double()
        expected_v = .999 * expected_v + .001 * gradient.double().square()
        a.step()
        b.step()
        torch.testing.assert_close(a.m[0], expected_m)
        torch.testing.assert_close(a.v[0], expected_v)
        torch.testing.assert_close(a.m[0], b.m[0])
        torch.testing.assert_close(a.v[0], b.v[0])
    assert not torch.allclose(p, q)


def test_unclipped_noiseless_adam_matches_reference_bias_correction():
    p = torch.nn.Parameter(torch.tensor([1., -2.]))
    reference = torch.nn.Parameter(p.detach().clone())
    opt = UCAdam([p], lr=.03, update_clip_norm=1e6, noise=FixedNoise([torch.zeros(2)]))
    ordinary = torch.optim.Adam([reference], lr=.03, betas=(.9, .999), eps=1e-8, weight_decay=0)
    for g in (torch.tensor([1., 3.]), torch.tensor([-2., .5]), torch.tensor([4., -1.])):
        p.grad, reference.grad = g.clone(), g.clone()
        stats = opt.step()
        ordinary.step()
        torch.testing.assert_close(p, reference)
        assert stats['clip_scale'] == 1 and stats['clip_indicator'] == 0


def test_nonprivate_probe_has_no_clipping_or_gaussian_draws(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Probe must not draw noise')
    monkeypatch.setattr(torch, 'randn', forbidden)
    p = torch.nn.Parameter(torch.tensor([1., -2.]))
    ref = torch.nn.Parameter(p.detach().clone())
    opt = UCAdam([p], lr=.001, update_clip_norm=None, noise=ZeroNoise([p], 250))
    ordinary = torch.optim.Adam([ref], lr=.001)
    for g in (torch.tensor([1., 3.]), torch.tensor([-2., .5])):
        p.grad = ref.grad = g
        stats = opt.step()
        ordinary.step()
        torch.testing.assert_close(p, ref)
        assert stats['clip_scale'] == 1 and stats['clip_indicator'] == 0
        assert stats['noise_marginal_std'] == stats['realized_noise_rms'] == 0


def test_large_finite_adam_state_has_finite_statistics_without_changing_update():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([1e20, -1e20])
    opt = UCAdam([p], lr=.001, update_clip_norm=1., noise=FixedNoise([torch.zeros(2)]))
    stats = opt.step()
    assert all(value is None or np.isfinite(value) for value in stats.values())
    assert np.isclose(stats['vhat_mean'], 1e40)
    assert np.isclose(stats['vhat_rms'], 1e40)
    torch.testing.assert_close(p, torch.tensor([-.001 / math.sqrt(2), .001 / math.sqrt(2)]))
    assert opt.m[0].dtype == opt.v[0].dtype == torch.float64


def test_four_microbatches_equal_raw_mean_gradient():
    torch.manual_seed(19)
    model = torch.nn.Linear(3, 2)
    inputs, targets = torch.randn(12, 3), torch.arange(12) % 2
    torch.nn.functional.cross_entropy(model(inputs), targets).backward()
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    for x, y in zip(inputs.chunk(4), targets.chunk(4)):
        backward_microbatch(model, x, y, 12)
    for p, g in zip(model.parameters(), expected):
        torch.testing.assert_close(p.grad, g)


def test_zero_direction_is_finite_and_still_adds_noise():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    p.grad = torch.zeros(2)
    opt = UCAdam([p], .1, 1., FixedNoise([torch.tensor([1., -2.])]))
    stats = opt.step()
    torch.testing.assert_close(p, torch.tensor([.9, 2.2]))
    assert stats['raw_update_norm'] == stats['clipped_update_norm'] == 0
    assert stats['clip_scale'] == 1 and stats['noise_signal_rms_ratio'] is None


def test_noise_lr_are_postprocessing_only():
    p, q = torch.nn.Parameter(torch.ones(8)), torch.nn.Parameter(torch.ones(8))
    na = TemporalGaussianNoise([p], [1.], 4., 2, 13)
    nb = TemporalGaussianNoise([q], [1.], 4., 2, 13)
    a, b = UCAdam([p], .01, 1., na), UCAdam([q], .03, 1., nb)
    p.grad = q.grad = torch.arange(1., 9.)
    sa, sb = a.step(), b.step()
    for key in ('raw_update_norm', 'clipped_update_norm', 'clip_scale', 'noise_marginal_std', 'adam_m_norm'):
        assert sa[key] == sb[key]
    torch.testing.assert_close((1 - q), 3 * (1 - p))
