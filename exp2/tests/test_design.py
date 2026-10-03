"""Mechanism math, local search, matched clipping and resume invariants."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from exp2.cells import (CELLS, CLIPS, LR_GRIDS, FAMILIES, SEARCH_SEED, FINAL_SEEDS,
    MATCHED_CLIPS, cell_for, first_search_trials, second_search_trials, third_search_trials,
    matched_search_trials, matched_final_trials, scale_reference_trials, trial_key, job)
from exp2.diagnostics import (adam_first_moment_matrix, prefix_accumulation_matrix,
    frozen_v_operator, frozen_v_noise_norms, frozen_v_diagnostics, CANCELLATION_BASELINE,
    PAPER_APPROX_METADATA, FROZEN_V_METADATA)
from exp2.train import ROOT, AugmentationTrace
from exp2 import sweep
from exp2.tests.test_experiment import configs


def results(jobs, score=.5):
    return [(j, Path(j['name']), dict(final=dict(test_top1=score), initialization_sha256='paired',
            epochs=[dict(epoch=e, clip_fraction=.25) for e in range(1, 6)])) for j in jobs]


@pytest.mark.parametrize('method', ['iid_scale', 'momentum_scale'])
@pytest.mark.parametrize('clip', CLIPS)
def test_stage3_local_neighbors_only_and_no_repeated_combinations(method, clip):
    first = first_search_trials()
    clips = {'iid_scale': clip, 'momentum_scale': clip}
    second = second_search_trials(clips)
    for lr in LR_GRIDS[method]:
        current = {m: job(m, lr if m == method else .002, clip) for m in clips}
        third = third_search_trials(current, first + second)
        old = {trial_key(j) for j in first + second}
        assert not old.intersection(trial_key(j) for j in third)
        assert len({trial_key(j) for j in third}) == len(third)
        clip_idx = CLIPS.index(clip)
        lr_idx = LR_GRIDS[method].index(lr)
        expected = {trial_key(job(method, learning_rate, c))
                    for c in CLIPS[max(0, clip_idx-1):clip_idx+2]
                    for learning_rate in LR_GRIDS[method][max(0, lr_idx-1):lr_idx+2]} - old
        assert {trial_key(j) for j in third if j['method'] == method} == expected
        assert all(j['seed'] == SEARCH_SEED and j['stage'] == 'stage3' for j in third)


def test_stage3_selection_global_not_just_local_or_stage2():
    first = results(first_search_trials())
    second = results(second_search_trials({'iid_scale': 200, 'momentum_scale': 200}))
    centers = {m: job(m, .003, 200) for m in ('iid_scale', 'momentum_scale')}
    third = results(third_search_trials(centers, [j for j, _, _ in first + second]))
    # A Stage 1 point outside the local neighborhood must still be eligible.
    iid = next(r for r in first if r[0]['method'] == 'iid_scale' and r[0]['max_grad_norm'] == 500)
    iid[2]['final']['test_top1'] = .95
    momentum = next(r for r in third if r[0]['method'] == 'momentum_scale')
    momentum[2]['final']['test_top1'] = .9
    selected = sweep.select_configs(first + second + third)
    assert selected['iid_scale']['max_grad_norm'] == 500
    assert selected['momentum_scale']['lr'] == momentum[0]['lr']
    assert selected['momentum_scale']['max_grad_norm'] == momentum[0]['max_grad_norm']
    # Complete accuracy ties prioritize clip, then lr across all stages.
    for _, _, summary in first + second + third:
        summary['final']['test_top1'] = .5
    tied = sweep.select_configs(first + second + third)
    assert tied['iid_scale']['max_grad_norm'] == 50
    assert tied['iid_scale']['lr'] == .002


def test_matched_search_same_lr_correct_mechanism_and_nine_new_finals():
    cfg = configs()
    cfg['iid_scale']['lr'] = .003
    cfg['momentum_scale']['lr'] = .007
    search = matched_search_trials(cfg)
    assert len(search) == 18
    assert all(j['seed'] == SEARCH_SEED for j in search)
    for family in FAMILIES:
        method = f'{family}_standard_matched'
        candidates = [j for j in search if j['method'] == method]
        assert tuple(j['max_grad_norm'] for j in candidates) == MATCHED_CLIPS
        assert all(j['lr'] == cfg[f'{family}_scale']['lr'] for j in candidates)
        assert cell_for(method) == CELLS[f'{family}_standard']
        assert cell_for(method)['noise'] == CELLS[f'{family}_scale']['noise']
    selected = sweep.select_matched_configs(results(search), dict.fromkeys(FAMILIES, .25))
    assert all(v['max_grad_norm'] == 1 for v in selected.values())
    finals = matched_final_trials(selected)
    assert len(finals) == 9
    assert all(cell_for(j['method'])['geometry'] == 'standard' for j in finals)
    assert {s: sum(j['seed'] == s for j in finals) for s in FINAL_SEEDS} == dict.fromkeys(FINAL_SEEDS, 3)
    reference = scale_reference_trials(cfg)
    assert len(reference) == 1 and reference[0]['method'] == 'prefix_scale'
    assert reference[0]['seed'] == SEARCH_SEED


def test_matched_selection_uses_epochs_3_5_mean_not_accuracy_or_early_epochs():
    search = results(matched_search_trials(configs()))
    targets = dict(iid=.2, prefix=.4, momentum=.6)
    for j, _, summary in search:
        family = j['method'].removesuffix('_standard_matched')
        clip = j['max_grad_norm']
        # C=3 has the exact target, even though its accuracy is lowest.
        late = targets[family] if clip == 3 else targets[family] + .125
        summary['epochs'] = [dict(epoch=e, clip_fraction=1. if e < 3 else late) for e in range(1, 6)]
        summary['final']['test_top1'] = 0. if clip == 3 else .99
    selected = sweep.select_matched_configs(search, targets)
    assert all(v['max_grad_norm'] == 3 for v in selected.values())
    assert len(sweep.matched_search_rows(search, targets)) == 18
    # Construct an exact binary tie at C=1 and C=3; smaller clip wins.
    for j, _, summary in search:
        family = j['method'].removesuffix('_standard_matched')
        for row in summary['epochs'][2:]:
            row['clip_fraction'] = .25 if j['max_grad_norm'] in (1, 3) else .75
    assert all(v['max_grad_norm'] == 1 for v in
               sweep.select_matched_configs(search, dict.fromkeys(FAMILIES, .25)).values())


def test_scale_target_follows_globally_selected_clip_and_lr():
    cfg = configs()
    points = [dict(job(f'{family}_scale', cfg[f'{family}_scale']['lr'],
                       cfg[f'{family}_scale']['max_grad_norm']), name=family) for family in FAMILIES]
    selected = results(points)
    for idx, (_, _, summary) in enumerate(selected):
        summary['epochs'] = [dict(epoch=e, clip_fraction=(e + idx) / 10) for e in range(1, 6)]
    targets = sweep.scale_targets(selected, cfg)
    assert targets == pytest.approx(dict(iid=.4, prefix=.5, momentum=.6))


def test_adam_A_bias_correction_and_prefix_L():
    beta = .9
    A = adam_first_moment_matrix(5, beta)
    L = prefix_accumulation_matrix(5)
    expected = np.array([[(1-beta)*beta**(t-j)/(1-beta**(t+1)) if j <= t else 0
                          for j in range(5)] for t in range(5)])
    np.testing.assert_array_equal(A, expected)
    np.testing.assert_allclose(A.sum(axis=1), 1.)
    assert A[0, 0] == pytest.approx(1.)
    np.testing.assert_array_equal(L, np.array([[int(j <= t) for j in range(5)] for t in range(5)]))


@pytest.mark.parametrize('beta1', [.0, .9, .95])
def test_frozen_operator_and_recurrence_match_explicit_small_matrices(beta1):
    rng = np.random.default_rng(42)
    T, Q = 6, 11
    p, s = np.exp(rng.normal(size=(2, T, Q)))
    M = np.tril(rng.normal(size=(T, T))) + np.eye(T)
    A = np.array([[(1-beta1)*beta1**(t-j)/(1-beta1**(t+1)) if j <= t else 0
                   for j in range(T)] for t in range(T)])
    L = np.tril(np.ones((T, T)))
    expected = [L @ np.diag(p[:, q]) @ A @ np.diag(1/s[:, q]) @ M for q in range(Q)]
    for q in range(Q):
        np.testing.assert_allclose(frozen_v_operator(p[:, q], s[:, q], M, beta1), expected[q], rtol=1e-13)
    norms, r_rms = frozen_v_noise_norms(p, s, M, beta1)
    np.testing.assert_allclose(norms, [np.linalg.norm(K, 'fro') for K in expected], rtol=1e-13)
    np.testing.assert_allclose(r_rms, np.sqrt(np.mean((p/s)**2, axis=0)), rtol=1e-14)
    diagnostics = frozen_v_diagnostics(p, s, M, 2.3, beta1)
    ratio = norms / (r_rms * np.linalg.norm(L @ A @ M, 'fro'))
    efficiency = norms / (r_rms * np.linalg.norm(L @ A @ (2.3*np.eye(T)), 'fro'))
    for quantile, label in ((.1, 'p10'), (.5, 'p50'), (.9, 'p90')):
        assert diagnostics['frozen_v_mf_distortion'][f'frozen_ratio_{label}'] == pytest.approx(np.quantile(ratio, quantile))
        assert diagnostics['mf_cancellation_efficiency'][label] == pytest.approx(np.quantile(efficiency, quantile))
    assert diagnostics['frozen_v_mf_distortion']['frozen_logabs_mean'] == pytest.approx(np.abs(np.log(ratio)).mean())
    assert diagnostics['frozen_v_mf_distortion']['frozen_logabs_median'] == pytest.approx(np.median(np.abs(np.log(ratio))))
    assert diagnostics['mf_cancellation_efficiency']['mean'] == pytest.approx(efficiency.mean())


def test_constant_p_and_s_frozen_distortion_one_iid_efficiency_one():
    p, s = np.full((8, 2048), 2.), np.full((8, 2048), 7.)
    M = 2.3 * np.eye(8)
    diag = frozen_v_diagnostics(p, s, M, 2.3)
    assert diag['frozen_v_mf_distortion']['frozen_ratio_p50'] == pytest.approx(1.)
    assert diag['frozen_v_mf_distortion']['frozen_logabs_mean'] == pytest.approx(0., abs=1e-14)
    assert diag['mf_cancellation_efficiency']['p50'] == pytest.approx(1.)
    # Constant per-coordinate p/s from individually time-constant p and s also
    # holds for MF. If p and s co-vary in time, A need not commute with diag(s).
    M = np.tril(np.ones((8, 8))) * 1.7
    assert frozen_v_diagnostics(p, s, M, 1.7)['frozen_v_mf_distortion']['frozen_ratio_p50'] == pytest.approx(1.)
    assert 'innovation_std_sum' in CANCELLATION_BASELINE and 'Lower' in CANCELLATION_BASELINE
    assert 'not exact Adam Jacobian' in FROZEN_V_METADATA
    assert PAPER_APPROX_METADATA == 'paper-style approximation; not exact Adam Jacobian'


def test_cancellation_baseline_uses_saved_innovation_scale():
    p, s, D = np.ones((5, 3)), np.ones((5, 3)), np.eye(5)
    original = frozen_v_diagnostics(p, s, D*2., 2.)
    rescaled = frozen_v_diagnostics(p, s, D*200., 200.)
    for key in original:
        assert original[key] == pytest.approx(rescaled[key], abs=1e-14)
    # Changing only the normalization (not actual M) changes efficiency.
    assert frozen_v_diagnostics(p, s, D*2., 4.)['mf_cancellation_efficiency']['p50'] == pytest.approx(.5)


def test_constant_ratio_with_covarying_p_s_keeps_specified_operator():
    s = np.array([[1.], [2.], [4.]])
    p = 3 * s
    M = np.eye(3)
    np.testing.assert_array_equal(p / s, np.full((3, 1), 3.))
    explicit = np.linalg.norm(frozen_v_operator(p[:, 0], s[:, 0], M), 'fro')
    ideal = 3 * np.linalg.norm(prefix_accumulation_matrix(3) @ adam_first_moment_matrix(3), 'fro')
    ratio = frozen_v_diagnostics(p, s, M, 1.)['frozen_v_mf_distortion']['frozen_ratio_p50']
    assert ratio == pytest.approx(explicit / ideal)
    assert ratio != pytest.approx(1.)


def test_rolling_augmentation_all_logical_batches_cpu_and_pairing():
    batches = [torch.arange(6*3*2*2, dtype=torch.float32).reshape(6, 3, 2, 2) + i for i in range(12)]
    digests = []
    for method in tuple(CELLS) + tuple(f'{f}_standard_matched' for f in FAMILIES):
        trace = AugmentationTrace(4)
        for inputs in batches:
            trace.update(inputs)
        digests.append(trace.hexdigest())
    assert len(set(digests)) == 1
    expected = hashlib.sha256()
    for i in (0, 4, 8):
        expected.update(batches[i][:4].numpy().tobytes())
    assert digests[0] == expected.hexdigest()
    changed = AugmentationTrace(4)
    for i, inputs in enumerate(batches):
        changed.update(inputs + 1 if i == 8 else inputs)
    assert changed.hexdigest() != digests[0]  # catches changes beyond epoch's first batch


def test_incomplete_resume_fails_before_launch_and_preserves_files(tmp_path, monkeypatch):
    values = dict(job('iid_scale', .002, 200), name='incomplete')
    directory = tmp_path / values['name']
    directory.mkdir()
    (directory / 'train.log').write_text('failed trial evidence')
    monkeypatch.setattr(sweep.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(RuntimeError) as exc:
        sweep.run_queue([values], tmp_path, ROOT / 'exp2/config.yaml')
    assert str(exc.value) == f'Incomplete trial directory: {directory}\nRemove it manually before restarting.'
    assert (directory / 'train.log').read_text() == 'failed trial evidence'
    assert list(directory.iterdir()) == [directory / 'train.log']
