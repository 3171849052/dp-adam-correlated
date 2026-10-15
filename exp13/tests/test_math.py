import numpy as np
import pytest
import torch
from exp13.math import (BETA1, BETA2, EPS, Evaluator, configure_torch, digest,
                        exp9_filter, innovations, inverse_strategy, materialize,
                        moment_matrix, momentum_initial, sensitivity_squared, target_mu)

configure_torch()


def recursive_moment(x, beta):
    state = np.zeros(x.shape[1], dtype=np.float64)
    result = []
    for t, row in enumerate(x, start=1):
        state = beta*state + (1-beta)*row
        result.append(state/(1-beta**t))
    return np.stack(result)


def explicit_workloads(a, M):
    L = np.tril(np.ones((len(M), len(M))))
    return [L @ np.diag(a[:, c]) @ M for c in range(a.shape[1])]


@pytest.mark.parametrize('beta', [BETA1, BETA2])
def test_bias_corrected_moment_matrix_matches_recursion(beta):
    x = np.random.default_rng(5).normal(size=(15, 6))
    np.testing.assert_allclose(moment_matrix(15, beta) @ x,
                               recursive_moment(x, beta), rtol=3e-14, atol=2e-15)


def test_inverse_and_fixed_participation_sensitivity():
    d, n = np.array([1., -.7, .15, -.02]), 15
    C = inverse_strategy(d, n)
    np.testing.assert_allclose(C @ materialize(d, n), np.eye(n), atol=3e-16)
    gram = C.T @ C
    reference = max(np.abs(gram[np.ix_(ix, ix)]).sum()
                    for j in range(n//5) for ix in [j + n//5*np.arange(5)])
    assert sensitivity_squared(C) == pytest.approx(reference, rel=2e-15)
    assert sensitivity_squared(np.eye(n)) == 5.


def test_gdp_calibration():
    from scipy.special import ndtr
    mu = target_mu()
    delta = ndtr(-8/mu+mu/2)-np.exp(8)*ndtr(-8/mu-mu/2)
    assert delta == pytest.approx(1e-5, rel=2e-12)


def test_gram_and_4x4_q_equal_explicit_coordinate_workloads():
    xi, _ = innovations(31, 10, 7, 'cpu')
    evaluator = Evaluator(xi)
    d = np.array([1., -.63, -.12, .06])
    result = evaluator.evaluate(d, details=True)
    a, M = result['a'].numpy(), evaluator.M.numpy()
    W = explicit_workloads(a, M)
    D = materialize(d, 10)
    explicit = result['sigma']**2 * sum(np.square(w @ D).sum() for w in W) / (10*7)
    assert result['J'] == pytest.approx(explicit, rel=3e-14)
    reference_Q = np.zeros((4, 4))
    for w in W:
        shifts = [w @ np.linalg.matrix_power(materialize([0., 1.], 10), i) for i in range(4)]
        for i in range(4):
            for j in range(4):
                reference_Q[i, j] += np.sum(shifts[i] * shifts[j]) / (10*7)
    np.testing.assert_allclose(result['Q'].numpy(), reference_Q, rtol=3e-14)
    assert np.linalg.eigvalsh(result['Q'].numpy()).min() > 0


def test_real_noise_squared_adam_and_frobenius_are_distinct():
    xi, _ = innovations(41, 15, 5, 'cpu')
    evaluator = Evaluator(xi)
    d = np.array([1., -.72, -.07, .02])
    metrics, result = evaluator.metrics(d)
    N = result['sigma'] * materialize(d, 15) @ xi.numpy()
    v = recursive_moment(N**2, BETA2)
    m = recursive_moment(N, BETA1)
    real = np.mean(np.cumsum(m / (np.sqrt(v)+EPS), axis=0)**2)
    np.testing.assert_allclose(result['noise'].numpy(), N, rtol=2e-14, atol=1e-17)
    np.testing.assert_allclose(result['vhat'].numpy(), v, rtol=3e-14)
    assert metrics['J_real'] == pytest.approx(real, rel=2e-14)
    assert not np.isclose(metrics['J'], metrics['J_real'], rtol=.01)


def test_expected_comparator_uses_time_varying_boundary_variance():
    xi, _ = innovations(51, 10, 5, 'cpu')
    evaluator = Evaluator(xi)
    d = np.array([1., -.7, -.1, .02])
    result = evaluator.evaluate(d, 'expected', details=True)
    variance = result['sigma']**2 * np.diag(materialize(d, 10) @ materialize(d, 10).T)
    expected = recursive_moment(variance[:, None], BETA2)
    np.testing.assert_allclose(result['vhat'].numpy(), expected, rtol=3e-14)
    a = np.repeat(result['a'].numpy(), 5, axis=1)
    np.testing.assert_allclose(evaluator.gram_q(torch.as_tensor(a)).numpy(), result['Q'].numpy(), rtol=2e-14)


def test_each_candidate_recomputes_vhat_q_and_keeps_xi_fixed():
    xi, sha = innovations(61, 10, 6, 'cpu')
    evaluator = Evaluator(xi)
    iid = np.array([1., 0., 0., 0.])
    d = np.array([1., -.8, .03, -.01])
    first = evaluator.evaluate(iid, details=True)
    changed = evaluator.evaluate(d, details=True)
    repeated = evaluator.evaluate(iid, details=True)
    assert evaluator.calls == 3
    assert digest(evaluator.xi.numpy()) == sha
    assert first['J'] == repeated['J']
    assert torch.equal(first['Q'], repeated['Q'])
    assert not torch.allclose(first['vhat'], changed['vhat'])
    assert not torch.allclose(first['Q'], changed['Q'])
    frozen = changed['sigma']**2 * float(d @ first['Q'].numpy() @ d)
    assert not np.isclose(frozen, changed['J'], rtol=.01)
    other, other_sha = innovations(62, 10, 6, 'cpu')
    assert sha != other_sha and not torch.equal(xi, other)


def test_exp9_initial_coefficients_and_original_objective():
    # Independent formal power-series identity: D(z)^2 W(z) = 1 through z^3.
    d = momentum_initial()
    w = np.cumsum(.9**np.arange(4))
    np.testing.assert_allclose(np.convolve(np.convolve(d, d), w)[:4], [1., 0., 0., 0.], atol=1e-15)
    fit, starts = exp9_filter(10)
    W = np.tril(np.ones((10, 10))) @ moment_matrix(10, .9) / .1
    direct = sensitivity_squared(inverse_strategy(fit, 10)) * np.mean(np.sum((W @ materialize(fit, 10))**2, axis=1))
    assert min(r['objective'] for r in starts) == pytest.approx(direct, rel=1e-12)


def test_fitting_is_reproducible_and_beats_both_initial_points():
    xi, sha = innovations(71, 10, 8, 'cpu')
    first = Evaluator(xi)
    d1, info1 = first.fit('actual')
    second = Evaluator(xi)
    d2, info2 = second.fit('actual')
    np.testing.assert_array_equal(d1, d2)
    assert info1 == info2
    assert sha == digest(first.xi.numpy())
    initial = min(first.evaluate(d) for d in (momentum_initial(), np.array([1., 0., 0., 0.])))
    assert first.evaluate(d1) <= initial*(1+1e-10)
    assert info1['evaluations'] == sum(row['evaluations'] for row in info1['starts'])


@pytest.mark.parametrize('n', [225, 250, 310])
def test_full_horizon_recursion_and_gram_with_few_coordinates(n):
    xi, _ = innovations(81, n, 3, 'cpu')
    evaluator = Evaluator(xi)
    d = np.array([1., -.8, -.05, .01])
    result = evaluator.evaluate(d, details=True)
    v = recursive_moment(result['noise'].numpy()**2, BETA2)
    np.testing.assert_allclose(result['vhat'].numpy(), v, rtol=5e-14)
    W = explicit_workloads(result['a'].numpy(), evaluator.M.numpy())
    direct = result['sigma']**2 * sum(np.square(w @ materialize(d, n)).sum() for w in W)/(n*3)
    assert result['J'] == pytest.approx(direct, rel=2e-12)


@pytest.mark.parametrize('gpu', [0, 1, 2, 3])
def test_gpu_float64_agrees_with_cpu(gpu):
    assert torch.cuda.device_count() >= 4
    cpu_xi, sha = innovations(91, 10, 16, 'cpu')
    gpu_xi, gpu_sha = innovations(91, 10, 16, f'cuda:{gpu}')
    assert sha == gpu_sha
    d = np.array([1., -.76, -.08, .02])
    cpu = Evaluator(cpu_xi).evaluate(d, details=True)
    evaluator = Evaluator(gpu_xi)
    result = evaluator.evaluate(d, details=True)
    assert result['Q'].dtype == torch.float64
    assert result['J'] == pytest.approx(cpu['J'], rel=3e-13)
    np.testing.assert_allclose(result['vhat'].cpu().numpy(), cpu['vhat'].numpy(), rtol=3e-14)
    assert evaluator.evaluate(d) == result['J']
