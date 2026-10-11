import builtins
import copy
import numpy as np
import pytest
import torch
from exp11 import RESULTS
from exp11.config import T, CLIP, SEED, INITIAL_LAMBDAS, protocol, save_json
from exp11.matrices import Objective, dense_components, get_matrix, weights
from exp11.bandinvmf import optimize_bias_filter, momentum_bias_workload
from exp11.noise import materialize, build_matrices, BandInvMFNoise
from exp11.privacy import calibrate, epsilon_from_mu, fixed_epoch_sensitivity


def test_complete_workload_and_weights():
    n = 12
    expected = np.zeros((n, n))
    for t in range(1, n + 1):
        for s in range(1, t + 1):
            expected[t-1, s-1] = sum(.9 ** (u-s) / (1-.9 ** u) for u in range(s, t+1))
    np.testing.assert_allclose(momentum_bias_workload(n, .9), expected, rtol=1e-14)
    w = weights(n)
    np.testing.assert_allclose(w.sum(axis=1), 1., atol=2e-14)
    assert np.count_nonzero(np.triu(w, 1)) == 0
    assert len(INITIAL_LAMBDAS) == 30 and INITIAL_LAMBDAS[1] == 1e-4 and INITIAL_LAMBDAS[-1] == 1e3


@pytest.mark.parametrize('d', [[1., 0., 0., 0.], [1., -.7, -.08, -.02], [1., .15, -.3, .08]])
def test_compact_equals_dense_including_startup(d):
    d = np.array(d)
    kernel = Objective()
    compact, dense = kernel.components(d), dense_components(d)
    for key in dense:
        assert compact[key] == pytest.approx(dense[key], rel=1e-11, abs=1e-22)
    if np.array_equal(d, [1., 0., 0., 0.]):
        assert compact['J2_bias_sq'] == pytest.approx(compact['sigma_avg'] ** 4, rel=1e-12)
        assert kernel.value(np.zeros(3), 2.7) == pytest.approx(3.7, rel=1e-13)


def test_lambda_zero_reproduces_exp9_and_gdp():
    d, S, W, meta = get_matrix(0.)
    initial = build_matrices('momentum_bandinvmf', T, 4, .9)[0]
    old_d, old = optimize_bias_filter(W, initial, spacing=45)
    assert meta['J1'] == pytest.approx(old['optimized_objective'], rel=1e-8)
    np.testing.assert_allclose(d, old_d, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(materialize(d, T) @ S, np.eye(T), atol=3e-12)
    gram = S.T @ S
    sensitivity = max(np.abs(gram[np.ix_(idx, idx)]).sum() for j in range(45)
                      for idx in [j + 45 * np.arange(5)]) ** .5
    assert sensitivity == pytest.approx(meta['sensitivity'], rel=1e-12)
    p = calibrate(S, protocol(0.))
    eps = epsilon_from_mu(CLIP * sensitivity / p['innovation_std_sum'], 1e-5)
    assert eps == pytest.approx(8., abs=1e-8)
    assert p['innovation_std_sum'] / 1000 == pytest.approx(meta['sigma_avg'], rel=1e-12)
    prefix = [epsilon_from_mu(CLIP * fixed_epoch_sensitivity(S, 5, 45, steps=t)
                             / p['innovation_std_sum'], 1e-5) for t in range(45, 226, 45)]
    assert all(a <= b + 1e-10 for a, b in zip(prefix, prefix[1:]))
    save_json(RESULTS / 'lambda_zero_privacy_check.json',
              dict(status='passed', exp9_objective=old['optimized_objective'], exp11_objective=meta['J1'],
                   exp9_coefficients=old_d.tolist(), exp11_coefficients=d.tolist(), epsilon=eps,
                   epoch_epsilons=prefix, **p))


def test_gaussian_square_mse_monte_carlo():
    d, _, _, meta = get_matrix(1., neighbors=[(0., get_matrix(0.)[0])])
    D, w = materialize(d, T), weights(T)
    rng = np.random.default_rng(911)
    total = np.zeros(T)
    total2 = np.zeros(T)
    draws = 80000
    for _ in range(draws // 2000):
        z = meta['sigma_avg'] * (D @ rng.standard_normal((T, 2000)))
        second = w @ z ** 2
        total += second.sum(axis=1)
        total2 += (second ** 2).sum(axis=1)
    empirical_mse = float((total2 / draws).mean())
    empirical_bias_sq = float(((total / draws) ** 2).mean())
    empirical_variance = float((total2 / draws - (total / draws) ** 2).mean())
    assert empirical_mse == pytest.approx(meta['J2_mse'], rel=.025)
    assert empirical_bias_sq == pytest.approx(meta['J2_bias_sq'], rel=.025)
    assert empirical_variance == pytest.approx(meta['J2_variance'], rel=.05)
    save_json(RESULTS / 'monte_carlo_check.json', dict(status='passed', draws=draws,
              analytic={k: meta[k] for k in ('J2_mse', 'J2_bias_sq', 'J2_variance')},
              empirical=dict(J2_mse=empirical_mse, J2_bias_sq=empirical_bias_sq, J2_variance=empirical_variance),
              lambda_value=1., simulation_only_math_validation=True))


def test_official_test_access_forbidden(monkeypatch):
    from exp11.data import prepare, cv_datasets
    original = builtins.open
    def checked(file, *args, **kwargs):
        assert 'cifar-100-python/test' not in str(file)
        return original(file, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', checked)
    prepare()
    training, validation, order = cv_datasets(dict(mean=(.5, .5, .5), std=(.5, .5, .5), crop_pct=.9))
    assert len(training) == len(order) == 45000 and len(validation) == 5000
    assert not set(order).intersection(validation.indices)
    image, label = validation[0]
    assert image.shape == (3, 224, 224) and 0 <= label < 100


def test_vit_exact_clipping_and_logical_adam():
    from exp11.train import seed_all
    from exp11.cv_model import ViTTiny
    from exp11.clipping import StandardGhostModule, LogicalBatch, clipped_microbatch
    seed_all()
    base = ViTTiny(patch_size=4, image_size=16, embed_dim=12, depth=1, heads=3, mlp_ratio=2, num_classes=3)
    C = 1.3
    model = StandardGhostModule(copy.deepcopy(base), C)
    opt = torch.optim.Adam(model.parameters(), lr=.003, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    ref_opt = torch.optim.Adam(base.parameters(), lr=.003, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    params = list(model.parameters())
    noise = BandInvMFNoise(params, [1., -.3, -.1, -.01], .2, T, SEED+1)
    reference_noise = BandInvMFNoise(list(base.parameters()), [1., -.3, -.1, -.01], .2, T, SEED+1)
    logical = LogicalBatch(model, opt, 2, 4, noise)
    for step in range(3):
        x, y = torch.randn(4, 3, 16, 16), torch.tensor([0, 1, 2, 1])
        totals = [torch.zeros_like(p) for p in base.parameters()]
        norms = []
        for xi, yi in zip(x, y):
            base.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(base(xi[None]), yi[None]).backward()
            norm = sum(p.grad.square().sum() for p in base.parameters()).sqrt()
            norms.append(norm)
            for total, p in zip(totals, base.parameters()):
                total.add_(p.grad * min(1., C/norm.item()))
        for start in (0, 2):
            logical.begin_microbatch()
            clipped_microbatch(model, x[start:start+2], y[start:start+2], C)
            torch.testing.assert_close(model.last_norms, torch.stack(norms[start:start+2]), rtol=5e-5, atol=3e-6)
            assert logical.finish_microbatch() == (start == 2)
        for p, total, innovation in zip(base.parameters(), totals, reference_noise.next()):
            p.grad = (total + innovation) / 4
        ref_opt.step()
        for p, q in zip(params, base.parameters()):
            torch.testing.assert_close(p, q, rtol=5e-5, atol=3e-6)
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(opt.state[p][key], ref_opt.state[q][key], rtol=5e-5, atol=3e-6)
        assert logical.optimizer_steps == noise.step_count == step+1
    model.remove_hooks()
