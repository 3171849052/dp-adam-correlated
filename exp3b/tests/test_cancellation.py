import copy
import hashlib

import numpy as np
import pytest
import torch
from exp3.bandinvmf import build_matrices, materialize
from exp3.model import ViTTiny
from exp3.optimizer import HybridOptimizer
from exp3b import offline_runtime
from exp3b.capture import SignalWriter, CapturedAdamBatch
from exp3b.replay import (AdamState, MuonState, PairedNoise, aggregate, momentum_theory,
                         replay_inputs, replay_one, run_replay, write_csv, FIELDS)
from exp3b.report import build, validate
from exp3b.spec import output_path, write_json
from exp3b.support import expected_names, support

offline_runtime()
torch.set_num_threads(1)


def toy_model():
    return ViTTiny(embed_dim=2, depth=12, heads=1, mlp_ratio=1, image_size=16)


def fixture_manifest(steps=250, method='momentum_standard'):
    names = expected_names()
    shapes = {name: [2, 2] for name in names}
    coefficients, _, _ = build_matrices('momentum_bandinvmf', 250, 4, .9)
    return dict(status='completed', method=method, shapes=shapes, support=list(names),
                dimension=48 * 4, steps=steps, coefficients=coefficients.tolist(),
                innovation_std_gradient=.13, actual_lr=.005, smoke=True)


def test_momentum_theory_matches_monte_carlo():
    steps = 12
    rows = momentum_theory(steps)
    coef, _, workload = build_matrices('momentum_bandinvmf', 250, 4, .9)
    d, w = materialize(coef, steps), materialize(workload[:steps], steps)
    z = np.random.default_rng(92).standard_normal((steps, 80000))
    empirical_mf = np.sqrt(np.mean((w @ d @ z) ** 2, axis=1))
    empirical_iid = np.sqrt(np.mean((w @ z) ** 2, axis=1))
    np.testing.assert_allclose(empirical_mf, [r['rmse_mf_mean'] for r in rows], rtol=.015)
    np.testing.assert_allclose(empirical_iid, [r['rmse_iid_mean'] for r in rows], rtol=.015)
    np.testing.assert_allclose(empirical_mf / empirical_iid,
                               [r['cancellation_mean'] for r in rows], rtol=.02)


def test_noise_pairing_is_the_exact_same_z():
    shapes = {'a': (3, 4), 'b': (2, 2)}
    coefficients = np.array([1., -.4, .2, -.1])
    noise = PairedNoise(shapes, coefficients, .7, 55, 'cpu')
    history = []
    for step in range(10):
        mf, iid = noise.next()
        for name in shapes:
            expected = iid[name] * coefficients[0]
            for lag in range(1, min(step + 1, len(coefficients))):
                expected.add_(history[-lag][name], alpha=float(coefficients[lag]))
            assert torch.equal(mf[name], expected)
        assert noise.history[0]['a'] is iid['a']
        history.append(iid)
    identical = PairedNoise(shapes, [1.], .7, 55, 'cpu')
    for saved in history:
        mf, iid = identical.next()
        assert all(torch.equal(mf[name], iid[name]) and torch.equal(iid[name], saved[name]) for name in shapes)


def test_adam_states_are_independent_and_match_actual_optimizer():
    shapes = {'a': (2, 3)}
    branches = [AdamState(shapes, 'cpu') for _ in range(3)]
    assert len({branch.m['a'].data_ptr() for branch in branches}) == 3
    assert len({branch.v['a'].data_ptr() for branch in branches}) == 3
    p = torch.nn.Parameter(torch.zeros(2, 3))
    optimizer = torch.optim.Adam([p], lr=.02, betas=(.9, .999), eps=1e-8)
    for step in range(1, 6):
        gradient = torch.arange(6).reshape(2, 3).float() * .04 + step * .1
        before = p.detach().clone()
        p.grad = gradient
        optimizer.step()
        update = branches[0].update({'a': gradient})['a']
        torch.testing.assert_close(update, (before - p) / .02, rtol=3e-6, atol=1e-6)
        branches[1].update({'a': gradient + .3})
        branches[2].update({'a': gradient - .4})
    assert not torch.equal(branches[0].m['a'], branches[1].m['a'])
    assert not torch.equal(branches[1].v['a'], branches[2].v['a'])
    before = branches[0].v['a'].clone()
    branches[1].v['a'].zero_()
    assert torch.equal(before, branches[0].v['a'])


def test_scale_signal_noise_only_and_no_clipping_bias():
    raw = torch.tensor([[40., -70.], [80., -10.]])
    s = torch.tensor([[2., 5.], [.5, 3.]])
    realized = raw * .01
    q = s * realized
    g = q / s
    tensors = {'a': dict(g=g, q=q, scale=s)}
    original = copy.deepcopy(tensors)
    zero = {'a': torch.zeros_like(g)}
    clean, mf, iid = replay_inputs(tensors, zero, zero, scaled=True)
    assert torch.equal(clean['a'], q / s)
    assert torch.equal(clean['a'], mf['a']) and torch.equal(clean['a'], iid['a'])
    states = [AdamState({'a': (2, 2)}, 'cpu') for _ in range(3)]
    for _ in range(3):
        updates = [branch.update(x)['a'] for branch, x in zip(states, (clean, mf, iid))]
        assert torch.equal(updates[0], updates[1]) and torch.equal(updates[0], updates[2])
    assert not torch.equal(raw, clean['a'])
    nmf, niid = {'a': torch.ones_like(g) * .7}, {'a': torch.ones_like(g) * -.2}
    clean, mf, iid = replay_inputs(tensors, nmf, niid, scaled=True)
    assert torch.equal(clean['a'], g)
    torch.testing.assert_close(mf['a'], g + nmf['a'] / s)
    torch.testing.assert_close(iid['a'], g + niid['a'] / s)
    assert all(torch.equal(tensors['a'][k], value) for k, value in original['a'].items())
    with pytest.raises(AssertionError):
        replay_inputs({'a': dict(tensors['a'], g=raw)}, zero, zero, scaled=True)


def test_muon_replay_matches_real_nesterov_ns5_shape_factor():
    torch.manual_seed(80)
    model = toy_model()
    optimizer = HybridOptimizer(model, .03, .01)
    selected = support(model)
    shadow = MuonState({n: p.shape for n, p in selected.items()}, 'cpu')
    assert shadow.beta == .95 and shadow.nesterov and shadow.ns_steps == 5
    for _ in range(3):
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        gradients = {n: p.grad.clone() for n, p in selected.items()}
        before = {n: p.detach().clone() for n, p in selected.items()}
        optimizer.step()
        updates = shadow.update(gradients)
        for name, p in selected.items():
            torch.testing.assert_close(updates[name], (before[name] - p) / .03, rtol=1e-5, atol=2e-6)
    _, _, workload = build_matrices('momentum_bandinvmf', 250, 4, .9)
    np.testing.assert_allclose(workload[:3], [1., 1.9, 2.71])


@pytest.mark.parametrize('method', ['momentum_standard', 'momentum_scale', 'mf_muon_standard'])
def test_exact_common_48_parameter_support_and_rmse(method):
    model = toy_model()
    selected = support(model)
    assert tuple(selected) == expected_names() and len(selected) == 48
    assert 'head.weight' not in selected and 'patch.weight' not in selected
    manifest = fixture_manifest(steps=2, method=method)
    gen = torch.Generator().manual_seed(27)
    tensors = {n: dict(g=torch.randn(2, 2, generator=gen)) for n in expected_names()}
    if method == 'momentum_scale':
        for saved in tensors.values():
            saved['scale'] = torch.ones(2, 2) * 2
            saved['q'] = saved['g'] * saved['scale']
            saved['g'] = saved['q'] / saved['scale']
    original = copy.deepcopy(tensors)
    rows = replay_one(manifest, lambda step: tensors, 73, 'cpu')
    assert all(np.isfinite(v) for row in rows for v in row.values())
    assert all(torch.equal(value, original[n][k]) for n, saved in tensors.items() for k, value in saved.items())
    # Independent reconstruction checks the denominator and complete first update.
    nmf, niid = PairedNoise(manifest['shapes'], manifest['coefficients'], .13, 73, 'cpu').next()
    branches = replay_inputs(tensors, nmf, niid, method == 'momentum_scale')
    state = MuonState if method == 'mf_muon_standard' else AdamState
    updates = [state(manifest['shapes'], 'cpu').update(x) for x in branches]
    expected = np.sqrt(sum(float((updates[1][n] - updates[0][n]).double().square().sum())
                           for n in expected_names()) / (48 * 4))
    assert rows[0]['rmse_mf'] == pytest.approx(expected, rel=1e-12)


def test_capture_pre_noise_scaled_signal_does_not_change_real_optimizer(tmp_path):
    model = toy_model()
    reference = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    ref_optimizer = torch.optim.Adam(reference.parameters(), lr=.005)
    selected = support(model)
    writer = SignalWriter(tmp_path / 'signals', selected, 'momentum_scale', {})
    for p in model.parameters():
        p._logical_scale = torch.ones_like(p) * 3.
    class Noise:
        def __init__(self, parameters):
            self.parameters = list(parameters)
        def next(self):
            return [torch.ones_like(p) * .4 for p in self.parameters]
    from exp2.scale import LogicalBatch
    captured = CapturedAdamBatch(model, optimizer, 1, 2, Noise(model.parameters()), scaled=True, writer=writer)
    ordinary = LogicalBatch(reference, ref_optimizer, 1, 2, Noise(reference.parameters()), scaled=True)
    captured.begin_microbatch()
    ordinary.begin_microbatch()
    gradients = []
    for p, rp in zip(model.parameters(), reference.parameters()):
        p.grad = torch.ones_like(p) * .03
        rp.grad = p.grad.clone()
        gradients.append(p.grad.clone())
    captured.finish_microbatch()
    ordinary.finish_microbatch()
    for p, rp in zip(model.parameters(), reference.parameters()):
        assert torch.equal(p, rp)
    saved = torch.load(tmp_path / 'signals/step_001.pt', weights_only=True)['tensors']
    for name, value in saved.items():
        assert torch.equal(value['g'], value['q'] / value['scale'])
        torch.testing.assert_close(value['g'], torch.ones_like(value['g']) * .015)


def test_250_step_outputs_finite_replay_preserves_signals_and_plot_exports(tmp_path):
    manifest = fixture_manifest()
    generator = torch.Generator().manual_seed(7)
    frozen = {n: dict(g=torch.randn(2, 2, generator=generator) * .2) for n in expected_names()}
    original = copy.deepcopy(frozen)
    samples = [replay_one(manifest, lambda step: frozen, seed, 'cpu') for seed in (81, 82)]
    assert all(torch.equal(saved['g'], original[n]['g']) for n, saved in frozen.items())
    theory = momentum_theory()
    for name, method in (('adam', 'MF-Adam'), ('adam_scale', 'MF-Adam-Scale'), ('muon', 'MF-Muon')):
        # Report fixture uses genuine 250-step samples; mechanism fidelity is
        # separately checked against actual Adam/Muon and scaled-path tests.
        rows = aggregate(samples, method, theory)
        validate(rows, method, 250)
        write_csv(tmp_path / f'{name}.csv', rows, FIELDS)
        write_json(tmp_path / f'{name}.json', dict(steps=250, support_count=48,
            support=list(expected_names()), dimension=192, replay_seeds=[81, 82],
            source_manifest=manifest))
    groups = build(tmp_path)
    assert all(len(rows) == 250 for rows in groups.values())
    assert (tmp_path / 'combined.csv').read_text().count('\n') == 1001
    for name in ('cumulative_rmse', 'cancellation_curve', 'relative_to_momentum'):
        for extension in ('png', 'pdf'):
            assert (tmp_path / f'{name}.{extension}').stat().st_size > 1000


def test_recorded_signal_checksums_and_replay_no_source_writes(tmp_path):
    model = toy_model()
    selected = support(model)
    writer = SignalWriter(tmp_path / 'source', selected, 'momentum_standard', dict(
        coefficients=[1., -.2, .1, -.05], innovation_std_gradient=.1,
        actual_lr=.005, smoke=True))
    parameters = list(model.parameters())
    totals = [torch.ones_like(p) * .2 for p in parameters]
    for step in (1, 2):
        writer.record(step, parameters, totals, 2)
    writer.finish(2)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path / 'source').iterdir()}
    run_replay(tmp_path / 'source', tmp_path / 'replayed', device='cpu', seeds=(72, 73), smoke=True)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path / 'source').iterdir()}
    assert before == after


def test_nonfinite_and_outside_outputs_fail(tmp_path):
    with pytest.raises(AssertionError):
        output_path('/tmp/outside-exp3b.json')
    rows = momentum_theory(2)
    rows[0]['rmse_mf_mean'] = float('nan')
    with pytest.raises(AssertionError):
        validate(rows, 'MF-Momentum', 2)
