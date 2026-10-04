import numpy as np
import pytest
import torch

from exp4.bandinvmf import build_matrices, materialize
from exp4.privacy import calibrate, full_temporal_sensitivity, gdp_delta, epsilon_from_mu
from exp4.mechanism import TemporalGaussianNoise
from exp4.optimizer_uc import UCAdam


def test_sgd_prefix_workload_and_known_four_band_coefficients():
    d, c, w = build_matrices(250, 4)
    np.testing.assert_array_equal(w, np.ones(250))
    np.testing.assert_array_equal(materialize(w, 250), np.tril(np.ones((250, 250))))
    np.testing.assert_allclose(d, [1., -.5, -.125, -.0625], atol=1e-12)
    np.testing.assert_allclose(materialize(d, 250) @ c, np.eye(250), atol=1e-12)
    assert c.min() >= 0
    np.testing.assert_allclose(full_temporal_sensitivity(c), np.sqrt((c.T @ c).sum()))


def test_identity_iid_gdp_is_250_independent_queries_with_2R():
    radius = 2.3
    _, c, _ = build_matrices(250, 4, identity=True)
    calibration = calibrate(c, radius)
    assert calibration['per_step_sensitivity'] == 2 * radius
    assert np.isclose(calibration['temporal_sensitivity'], np.sqrt(250))
    assert np.isclose(calibration['innovation_std'], 2 * radius * np.sqrt(250) / calibration['target_mu'])
    assert np.isclose(gdp_delta(calibration['target_mu'], 8.), 1e-5)
    assert np.isclose(epsilon_from_mu(calibration['target_mu'], 1e-5), 8.)
    assert calibration['sampling_amplification'] is False
    assert calibration['batch_size_division'] is False


def test_full_temporal_bound_covers_arbitrary_changing_directions_and_prefixes():
    _, c, _ = build_matrices(250, 4)
    calibration = calibrate(c, 1.7)
    rng = np.random.default_rng(81)
    delta = rng.normal(size=(250, 5))
    delta *= 3.4 / np.linalg.norm(delta, axis=1, keepdims=True)
    assert np.linalg.norm(c @ delta) <= calibration['trajectory_sensitivity']
    aligned = np.zeros_like(delta)
    aligned[:, 0] = 3.4
    assert np.isclose(np.linalg.norm(c @ aligned), calibration['trajectory_sensitivity'])
    prefix = [full_temporal_sensitivity(c, steps=t) for t in (1, 50, 100, 150, 200, 250)]
    assert prefix == sorted(prefix)
    assert np.isclose(prefix[-1], calibration['temporal_sensitivity'])
    # Identity full participation counts every step, rather than five epochs.
    assert np.isclose(full_temporal_sensitivity(np.eye(250)), np.sqrt(250))


def test_signed_strategy_uses_safe_row_absolute_sum_for_adaptive_queries():
    c = np.array([[1., 0., 0.], [-2., 1., 0.], [1., -1., 1.]])
    assert np.isclose(full_temporal_sensitivity(c), np.sqrt(1 + 9 + 9))


@pytest.mark.parametrize('identity', [False, True])
def test_bandwidth_one_or_identity_matches_iid_noise_and_uc_trajectory(identity):
    d, c, _ = build_matrices(10, 1 if not identity else 4, identity=identity)
    np.testing.assert_array_equal(d, [1.])
    np.testing.assert_array_equal(c, np.eye(10))
    p, q = torch.nn.Parameter(torch.ones(11)), torch.nn.Parameter(torch.ones(11))
    sigma = calibrate(c, .8)['innovation_std']
    na = TemporalGaussianNoise([p], [1.], sigma, 10, 71)
    nb = TemporalGaussianNoise([q], d, sigma, 10, 71)
    a, b = UCAdam([p], .003, .8, na), UCAdam([q], .003, .8, nb)
    for step in range(10):
        p.grad = q.grad = torch.sin(torch.arange(11.) + step)
        assert a.step() == b.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(a.m[0], b.m[0], rtol=0, atol=0)
    assert na.history == nb.history == []


def test_filter_uses_innovations_not_previous_correlated_outputs():
    d, _, _ = build_matrices(6, 4)
    p = torch.nn.Parameter(torch.zeros(7))
    sigma, seed = 1.2, 199
    noise = TemporalGaussianNoise([p], d, sigma, 6, seed)
    generator = torch.Generator().manual_seed(seed)
    innovations = []
    for t in range(6):
        innovations.append(torch.randn(7, generator=generator) * sigma)
        expected = innovations[t].clone()
        for lag in range(1, min(t + 1, 4)):
            expected.add_(innovations[t - lag], alpha=float(d[lag]))
        torch.testing.assert_close(noise.next()[0], expected, rtol=0, atol=0)
        assert np.isclose(noise.marginal_std(t), sigma * np.linalg.norm(d[:min(t + 1, 4)]))
        assert len(noise.history) <= 3
    with pytest.raises(AssertionError):
        noise.next()


def test_empirical_temporal_covariance_matches_DDt():
    d, _, _ = build_matrices(6, 4)
    p = torch.nn.Parameter(torch.zeros(50000))
    noise = TemporalGaussianNoise([p], d, 1., 6, 52)
    samples = np.stack([noise.next()[0].numpy() for _ in range(6)])
    D = materialize(d, 6)
    np.testing.assert_allclose(np.cov(samples), D @ D.T, atol=.025, rtol=.035)
