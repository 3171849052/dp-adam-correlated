"""Four single-GPU workers; three search barriers, matched clipping, paired finals."""
import argparse
from collections import deque
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import yaml
from exp2.cells import (CELLS, ANCHORS, FINAL_SEEDS, FAMILIES, cell_for,
                        first_search_trials, second_search_trials, third_search_trials,
                        final_trials, matched_search_trials, matched_final_trials,
                        scale_reference_trials, trial_key)
from exp2.diagnostics import (save_json, write_csv, mean_std, factorial_effects,
                             paper_approx_mf_distortion, frozen_v_diagnostics,
                             PAPER_APPROX_METADATA, FROZEN_V_METADATA, CANCELLATION_BASELINE)
from exp2.train import ROOT, load_config
from exp2.model import checkpoint_path, checkpoint_sha256

GPUS = (0, 1, 2, 3)
TRIAL_FILES = ('summary.json', 'config.yaml', 'train.log', 'metrics.csv', 'train_order.npy',
               'final.pt', 'matrices.npz', 'mechanism_trace.npz', 'mechanism_metrics.csv')


def incomplete(directory):
    raise RuntimeError(f'Incomplete trial directory: {directory}\nRemove it manually before restarting.')


def read_completed(directory, job, config_path, smoke=False):
    if not all((directory / filename).is_file() for filename in TRIAL_FILES):
        incomplete(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    if summary['status'] != 'completed':
        incomplete(directory)
    assert summary['smoke'] == smoke
    for key in ('method', 'seed', 'lr', 'max_grad_norm', 'eps_scale', 'num_bands'):
        assert summary[key] == job.get(key), (directory, key)
    for key, value in cell_for(job['method']).items():
        assert summary[key] == value
    expected_config = load_config(config_path)
    expected_config['seed'] = job['seed']
    expected_config['optimizer']['lr'] = job['lr']
    expected_config['privacy']['max_grad_norm'] = job['max_grad_norm']
    resolved = yaml.safe_load((directory / 'config.yaml').read_text())
    for key, value in expected_config.items():
        assert resolved[key] == value, (directory, key)
    assert summary['planned_total_steps'] == 250
    assert summary['pretrained_checkpoint_sha256'] == checkpoint_sha256(checkpoint_path())
    steps, epochs = (1, 1) if smoke else (250, 5)
    assert summary['optimizer_steps'] == summary['noise_steps'] == steps
    assert summary['physical_batches'] == steps * 4
    assert len(summary['epochs']) == len(summary['augmentation_trace_sha256']) == epochs
    with (directory / 'metrics.csv').open() as f:
        metrics = list(csv.DictReader(f))
    assert len(metrics) == epochs
    assert [row['augmentation_trace_sha256'] for row in metrics] == summary['augmentation_trace_sha256']
    with np.load(directory / 'mechanism_trace.npz') as trace:
        assert trace['p_trace'].shape == trace['r_trace'].shape == trace['s_trace'].shape == (steps, 2048)
        assert trace['coordinate_indices'].shape == (2048,)
        for key in ('p_trace', 's_trace', 'r_trace'):
            assert np.isfinite(trace[key]).all() and (trace[key] > 0).all()
        np.testing.assert_allclose(trace['r_trace'], trace['p_trace'] / trace['s_trace'], rtol=1e-6)
    return summary


def run_queue(jobs, base, config_path, smoke=False):
    """Reuse validated completed trials; any failure stops before the next stage."""
    pending, active, results = deque(), {}, []
    for job in jobs:
        directory = base / job['name']
        if directory.exists():
            results.append((job, directory, read_completed(directory, job, config_path, smoke)))
        else:
            pending.append(job)
    failed, completed = False, []
    while pending or active:
        for gpu in GPUS:
            if gpu in active:
                process, stream, job, directory = active[gpu]
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del active[gpu]
                print(f'{directory}: exit {code}', flush=True)
                if code:
                    failed = True
                else:
                    completed.append((job, directory))
            if pending:
                job = pending.popleft()
                directory = base / job['name']
                directory.mkdir(parents=True)
                stream = (directory / 'train.log').open('x')
                cmd = [sys.executable, '-u', '-m', 'exp2.train', '--config', str(config_path),
                       '--method', job['method'], '--seed', str(job['seed']), '--lr', str(job['lr']),
                       '--max-grad-norm', str(job['max_grad_norm']), '--result-dir', str(directory)]
                if 'eps_scale' in job:
                    cmd += ['--eps-scale', str(job['eps_scale'])]
                if smoke:
                    cmd += ['--smoke']
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                           TMPDIR=str(ROOT / 'exp2/cache/tmp'))
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT)
                active[gpu] = process, stream, job, directory
                print(f'GPU {gpu}: {directory}', flush=True)
        if active:
            time.sleep(.1)
    if failed:
        raise RuntimeError('Trial failed; see train.log. Pipeline stopped before next stage.')
    # Drain every worker before validating outputs, so invalid artifacts cannot
    # leave other GPU workers running after this stage fails.
    results.extend((job, directory, read_completed(directory, job, config_path, smoke))
                   for job, directory in completed)
    return results


def result_rows(results):
    return [dict(stage=job['stage'], **{k: job.get(k) for k in
                 ('method', 'seed', 'lr', 'eps_scale', 'max_grad_norm', 'num_bands')},
                 **cell_for(job['method']), final_test_top1=summary['final']['test_top1'],
                 initialization_sha256=summary['initialization_sha256'], result_dir=str(directory))
            for job, directory, summary in sorted(results, key=lambda item: item[0]['name'])]


def best(results):
    return min(results, key=lambda item: (-item[2]['final']['test_top1'],
                                         item[0]['max_grad_norm'], item[0]['lr']))


def select_configs(results):
    """Freeze only after Stage 3; consider ALL Stage 1+2+3 points per method."""
    configs = {method: dict(values) for method, values in ANCHORS.items()}
    for method in ('prefix_standard', 'iid_scale', 'momentum_scale'):
        winner = best([r for r in results if r[0]['method'] == method])[0]
        configs[method] = {k: winner[k] for k in ('lr', 'max_grad_norm', 'eps_scale', 'num_bands')
                           if k in winner}
    return {method: configs[method] for method in CELLS}


def late_clip_fraction(summary):
    epochs = {row['epoch']: row for row in summary['epochs']}
    assert set(epochs) == {1, 2, 3, 4, 5}
    return float(np.mean([epochs[epoch]['clip_fraction'] for epoch in (3, 4, 5)]))


def scale_targets(results, configs):
    targets = {}
    for family in FAMILIES:
        method = f'{family}_scale'
        cfg = configs[method]
        key = trial_key(dict(cfg, method=method, seed=20261001))
        matches = [s for j, _, s in results if trial_key(j) == key]
        assert len(matches) == 1
        targets[family] = late_clip_fraction(matches[0])
    return targets


def select_matched_configs(results, targets):
    configs = {}
    for family in FAMILIES:
        method = f'{family}_standard_matched'
        candidates = [r for r in results if r[0]['method'] == method]
        winner = min(candidates, key=lambda r: (abs(late_clip_fraction(r[2]) - targets[family]),
                                                r[0]['max_grad_norm']))[0]
        configs[method] = {k: winner[k] for k in ('lr', 'max_grad_norm', 'num_bands') if k in winner}
    return configs


def matched_search_rows(results, targets):
    rows = result_rows(results)
    by_directory = {str(d): s for _, d, s in results}
    for row in rows:
        family = row['method'].removesuffix('_standard_matched')
        measured = late_clip_fraction(by_directory[row['result_dir']])
        row.update(target_clip_fraction=targets[family],
                   mean_clip_fraction_epochs_3_5=measured,
                   clip_fraction_absolute_difference=abs(measured - targets[family]))
    return rows


def verify_pairing(results, expected_methods=None):
    """Check saved artifacts across factorial and matched methods, within seed."""
    expected_methods = set(CELLS) if expected_methods is None else set(expected_methods)
    assert len({s['pretrained_checkpoint_sha256'] for _, _, s in results}) == 1
    with np.load(results[0][1] / 'mechanism_trace.npz') as trace:
        expected_indices = trace['coordinate_indices']
    for seed in sorted({j['seed'] for j, _, _ in results}):
        paired = [(j, d, s) for j, d, s in results if j['seed'] == seed]
        assert {j['method'] for j, _, _ in paired} == expected_methods
        assert len(paired) == len(expected_methods)
        for key in ('initialization_sha256', 'classifier_initialization_sha256'):
            assert len({s[key] for _, _, s in paired}) == 1
        assert len({tuple(s['augmentation_trace_sha256']) for _, _, s in paired}) == 1
        order = np.load(paired[0][1] / 'train_order.npy')
        conventions = set()
        for _, directory, _ in paired:
            np.testing.assert_array_equal(order, np.load(directory / 'train_order.npy'))
            with np.load(directory / 'mechanism_trace.npz') as trace:
                np.testing.assert_array_equal(expected_indices, trace['coordinate_indices'])
            cfg = yaml.safe_load((directory / 'config.yaml').read_text())
            conventions.add((cfg['augmentation_rng_convention'], cfg['num_workers']))
        assert len(conventions) == 1
        for family in FAMILIES:
            directories = [d for j, d, _ in paired if j['method'].startswith(f'{family}_')]
            with np.load(directories[0] / 'matrices.npz') as reference:
                for directory in directories[1:]:
                    with np.load(directory / 'matrices.npz') as other:
                        for key in ('noising_coefficients', 'strategy', 'workload_coefficients', 'W'):
                            np.testing.assert_array_equal(reference[key], other[key])
                        # Only clip-dependent GDP scalar changes; D stays identical.
                        np.testing.assert_allclose(reference['M'] / reference['innovation_std_sum'],
                            other['M'] / other['innovation_std_sum'], rtol=1e-14, atol=1e-15)


def effects_json(effects, comparison):
    return dict(comparison=comparison, units='absolute top1 fraction', paired=effects,
        aggregate={key: mean_std([r[key] for r in effects]) for key in effects[0] if key != 'seed'})


def aggregate_final(results, configs, base):
    assert len(results) == 18
    verify_pairing(results)
    write_csv(base / 'final_multiseed.csv', result_rows(results))
    effects = []
    for seed in FINAL_SEEDS:
        accuracy = {j['method']: s['final']['test_top1'] for j, _, s in results if j['seed'] == seed}
        effects.append(dict(seed=seed, **factorial_effects(accuracy)))
    write_csv(base / 'factorial_effects.csv', [dict(comparison='tuned_system_effects', **row) for row in effects])
    save_json(base / 'factorial_effects.json', effects_json(effects, 'tuned_system_effects'))
    final_summary = {}
    for method in CELLS:
        trials = sorted((j['seed'], s) for j, _, s in results if j['method'] == method)
        assert [seed for seed, _ in trials] == list(FINAL_SEEDS)
        top1 = [s['final']['test_top1'] for _, s in trials]
        stats = mean_std(top1)
        final_summary[method] = dict(chosen_hyperparameters=configs[method], seeds=list(FINAL_SEEDS),
            top1_each_seed=top1, top1_mean=stats['mean'], top1_std=stats['sample_std'])
    save_json(base / 'final_summary.json', dict(comparison='tuned_system_effects', cells=final_summary))
    aggregate_mechanisms(results, base)


def aggregate_matched(final, matched, configs, targets, base):
    assert len(final) == 18 and len(matched) == 9
    methods = set(CELLS) | {f'{family}_standard_matched' for family in FAMILIES}
    verify_pairing(final + matched, methods)
    scale = [r for r in final if cell_for(r[0]['method'])['geometry'] == 'scale']
    rows = result_rows(scale + matched)
    for row in rows:
        row['comparison'] = 'matched-clipping geometry comparison'
    write_csv(base / 'matched_multiseed.csv', rows)
    effects = []
    for seed in FINAL_SEEDS:
        accuracy = {j['method'].removesuffix('_matched'): s['final']['test_top1']
                    for j, _, s in scale + matched if j['seed'] == seed}
        effects.append(dict(seed=seed, **{key + '_matched': value for key, value in
                                         factorial_effects(accuracy).items()}))
    write_csv(base / 'matched_effects.csv', [dict(comparison='matched-clipping geometry comparison', **row)
                                           for row in effects])
    output = effects_json(effects, 'matched-clipping geometry comparison')
    output.update(standard_configs=configs, target_clip_fractions=targets,
                  matching_seed=20261001, matching_epochs=[3, 4, 5],
                  interpretation='reduces tuning and clipping saturation confounding; not a strict causal geometry effect',
                  scale_trials_reused=True)
    save_json(base / 'matched_effects.json', output)
    aggregate_mechanisms(final + matched, base)


def aggregate_mechanisms(results, base):
    mechanism_summary, papers, frozens, cancellations = {}, [], [], []
    for method in sorted({j['method'] for j, _, _ in results}):
        trials = sorted((j['seed'], d, s) for j, d, s in results if j['method'] == method)
        stats = mean_std([s['final']['test_top1'] for _, _, s in trials])
        per_trial, per_paper, per_frozen, per_cancellation = [], [], [], []
        for seed, directory, _ in trials:
            with (directory / 'mechanism_metrics.csv').open() as f:
                metrics = list(csv.DictReader(f))
            per_trial.append({key: float(np.mean([float(row[key]) for row in metrics if row[key] != '']))
                              for key in ('r_cv', 'r_anisotropy', 'temporal_drift')})
            if cell_for(method)['noise'] != 'iid':
                with np.load(directory / 'mechanism_trace.npz') as trace, \
                     np.load(directory / 'matrices.npz') as matrices:
                    paper = paper_approx_mf_distortion(trace['r_trace'], matrices['M'], matrices['W'])
                    cfg = yaml.safe_load((directory / 'config.yaml').read_text())
                    diagnostics = frozen_v_diagnostics(trace['p_trace'], trace['s_trace'], matrices['M'],
                        float(matrices['innovation_std_sum']), cfg['optimizer']['beta1'])
                frozen = diagnostics['frozen_v_mf_distortion']
                cancellation = diagnostics['mf_cancellation_efficiency']
                papers.append(dict(method=method, seed=seed, **paper))
                frozens.append(dict(method=method, seed=seed, **frozen))
                cancellations.append(dict(method=method, seed=seed, **cancellation))
                per_paper.append(paper)
                per_frozen.append(frozen)
                per_cancellation.append(cancellation)
        entry = dict(top1_mean=stats['mean'], top1_std=stats['sample_std'],
            **{key + '_mean': float(np.mean([r[key] for r in per_trial])) for key in per_trial[0]},
            paper_approx_ratio_median=None, paper_approx_logabs_mean=None,
            frozen_v_ratio_median=None, frozen_v_logabs_mean=None,
            mf_cancellation_efficiency_median=None)
        if per_paper:
            entry.update(paper_approx_ratio_median=float(np.median([r['paper_approx_ratio_p50'] for r in per_paper])),
                         paper_approx_logabs_mean=float(np.mean([r['paper_approx_logabs_mean'] for r in per_paper])),
                         frozen_v_ratio_median=float(np.median([r['frozen_ratio_p50'] for r in per_frozen])),
                         frozen_v_logabs_mean=float(np.mean([r['frozen_logabs_mean'] for r in per_frozen])),
                         mf_cancellation_efficiency_median=float(np.median([r['p50'] for r in per_cancellation])))
        mechanism_summary[method] = entry
    write_csv(base / 'paper_approx_mf_distortion.csv', papers)
    write_csv(base / 'frozen_v_mf_distortion.csv', frozens)
    write_csv(base / 'mf_cancellation_efficiency.csv', cancellations)
    save_json(base / 'mechanism_summary.json', dict(cells=mechanism_summary,
        pairs={family: [f'{family}_standard', f'{family}_scale'] for family in FAMILIES},
        matched_pairs={family: [f'{family}_standard_matched', f'{family}_scale'] for family in FAMILIES},
        diagnostic_scope='2048 fixed sampled coordinates; full observed trajectory and saved M',
        paper_approx_mf_distortion=PAPER_APPROX_METADATA,
        frozen_v_mf_distortion=FROZEN_V_METADATA,
        mf_cancellation_efficiency_baseline=CANCELLATION_BASELINE,
        aggregation='equal trial means of step metrics; drift excludes first step; ratios are median of trial medians',
        top1_units='fraction', top1_std='sample std across three seeds (ddof=1)'))


def freeze(path, configs):
    if path.exists():
        assert json.loads(path.read_text()) == configs, f'Frozen configs differ: {path}'
    else:
        save_json(path, configs)
    return json.loads(path.read_text())


def main(args):
    assert torch.cuda.device_count() >= 4, 'GPUs 0,1,2,3 required'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '0,1,2,3')
    load_config(args.config)
    checkpoint_path()
    base = ROOT / 'exp2/results'
    base.mkdir(parents=True, exist_ok=True)
    (ROOT / 'exp2/cache/tmp').mkdir(parents=True, exist_ok=True)
    if args.smoke:
        configs = dict(ANCHORS, iid_scale=dict(lr=2e-3, max_grad_norm=200., eps_scale=.1),
                       prefix_standard=dict(lr=2e-3, max_grad_norm=1., num_bands=4),
                       momentum_scale=dict(lr=2e-3, max_grad_norm=200., eps_scale=.1, num_bands=4))
        jobs = [j for j in final_trials(configs) if j['seed'] == FINAL_SEEDS[0]]
        results = run_queue(jobs, base / 'smoke', args.config, smoke=True)
        verify_pairing(results)
        save_json(base / 'smoke_summary.json', dict(status='passed', cells=6,
            logical_steps_per_cell=1, physical_batches_per_cell=4, physical_batch_size=250,
            paired_initialization_order_augmentation_and_coordinates=True, pipeline_stage='smoke only'))
        return 0
    first = run_queue(first_search_trials(), base / 'search', args.config)
    write_csv(base / 'search_stage1_summary.csv', result_rows(first))
    clip_winners = {method: best([r for r in first if r[0]['method'] == method])
                    for method in ('iid_scale', 'momentum_scale')}
    second = run_queue(second_search_trials({m: r[0]['max_grad_norm'] for m, r in clip_winners.items()}),
                       base / 'search', args.config)
    assert len(first) == 16 and len(second) == 7
    reused = [(dict(j, stage='stage2_reused'), d, s) for j, d, s in clip_winners.values()]
    write_csv(base / 'search_stage2_summary.csv', result_rows(second + reused))
    # Local grid center is the winner among Stage 2's fixed-clip LR grid only.
    current = {method: best([r for r in second + reused if r[0]['method'] == method])[0]
               for method in clip_winners}
    third = run_queue(third_search_trials(current, [j for j, _, _ in first + second]),
                      base / 'search', args.config)
    write_csv(base / 'search_stage3_summary.csv', result_rows(third))
    all_search = first + second + third
    write_csv(base / 'search_summary.csv', result_rows(all_search))
    configs = freeze(base / 'selected_configs.json', select_configs(all_search))
    # Prefix Scale is a fixed anchor, but its epochs 3-5 target needs a search-seed run.
    reference = run_queue(scale_reference_trials(configs), base / 'search', args.config)
    targets = scale_targets(all_search + reference, configs)
    matched_search = run_queue(matched_search_trials(configs), base / 'matched_search', args.config)
    write_csv(base / 'matched_search_summary.csv', matched_search_rows(matched_search, targets))
    matched_configs = freeze(base / 'matched_configs.json', select_matched_configs(matched_search, targets))
    final = run_queue(final_trials(configs), base / 'final', args.config)
    aggregate_final(final, configs, base)
    matched = run_queue(matched_final_trials(matched_configs), base / 'matched_final', args.config)
    aggregate_matched(final, matched, matched_configs, targets, base)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'exp2/config.yaml')
    parser.add_argument('--smoke', action='store_true')
    sys.exit(main(parser.parse_args()))
