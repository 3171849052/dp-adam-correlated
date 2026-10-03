"""Four single-GPU workers; two search barriers, frozen selection, paired finals."""
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
from exp2.cells import (CELLS, ANCHORS, FINAL_SEEDS, first_search_trials,
                        second_search_trials, final_trials)
from exp2.diagnostics import (save_json, write_csv, mean_std, factorial_effects,
                             mf_distortion)
from exp2.train import ROOT, load_config
from exp2.model import checkpoint_path, checkpoint_sha256

GPUS = (0, 1, 2, 3)


def read_completed(directory, job, config_path, smoke=False):
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['status'] == 'completed' and summary['smoke'] == smoke
    for key in ('method', 'seed', 'lr', 'max_grad_norm', 'eps_scale', 'num_bands'):
        assert summary[key] == job.get(key), (directory, key)
    for key, value in CELLS[job['method']].items():
        assert summary[key] == value
    for filename in ('config.yaml', 'train.log', 'metrics.csv', 'train_order.npy',
                     'final.pt', 'matrices.npz', 'mechanism_trace.npz', 'mechanism_metrics.csv'):
        assert (directory / filename).is_file(), (directory, filename)
    expected_config = load_config(config_path)
    expected_config['seed'] = job['seed']
    expected_config['optimizer']['lr'] = job['lr']
    expected_config['privacy']['max_grad_norm'] = job['max_grad_norm']
    resolved = yaml.safe_load((directory / 'config.yaml').read_text())
    for key, value in expected_config.items():
        assert resolved[key] == value, (directory, key)
    assert summary['planned_total_steps'] == 250
    assert summary['pretrained_checkpoint_sha256'] == checkpoint_sha256(checkpoint_path())
    assert summary['optimizer_steps'] == summary['noise_steps'] == (1 if smoke else 250)
    with np.load(directory / 'mechanism_trace.npz') as trace:
        assert trace['r_trace'].shape == trace['s_trace'].shape == ((1 if smoke else 250), 2048)
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
    failed = False
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
                    results.append((job, directory, read_completed(directory, job, config_path, smoke)))
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
    return results


def result_rows(results):
    return [dict(stage=job['stage'], **{k: job.get(k) for k in
                 ('method', 'seed', 'lr', 'eps_scale', 'max_grad_norm', 'num_bands')},
                 **CELLS[job['method']], final_test_top1=summary['final']['test_top1'],
                 initialization_sha256=summary['initialization_sha256'], result_dir=str(directory))
            for job, directory, summary in sorted(results, key=lambda item: item[0]['name'])]


def best(results, key):
    return min(results, key=lambda item: (-item[2]['final']['test_top1'], item[0][key]))


def select_configs(first, second):
    configs = {method: dict(values) for method, values in ANCHORS.items()}
    configs['prefix_standard'] = {key: value for key, value in
        best([r for r in first if r[0]['method'] == 'prefix_standard'], 'lr')[0].items()
        if key in ('lr', 'max_grad_norm', 'num_bands')}
    for method in ('iid_scale', 'momentum_scale'):
        clip_winner = best([r for r in first if r[0]['method'] == method], 'max_grad_norm')
        candidates = [clip_winner] + [r for r in second if r[0]['method'] == method]
        winner = best(candidates, 'lr')[0]
        configs[method] = {k: winner[k] for k in ('lr', 'max_grad_norm', 'eps_scale', 'num_bands')
                           if k in winner}
    return {method: configs[method] for method in CELLS}


def verify_pairing(results):
    """Check actual saved artifacts rather than relying only on RNG promises."""
    assert len({s['pretrained_checkpoint_sha256'] for _, _, s in results}) == 1
    expected_indices = np.load(results[0][1] / 'mechanism_trace.npz')['coordinate_indices']
    for seed in sorted({j['seed'] for j, _, _ in results}):
        paired = [(j, d, s) for j, d, s in results if j['seed'] == seed]
        assert {j['method'] for j, _, _ in paired} == set(CELLS)
        for key in ('initialization_sha256', 'classifier_initialization_sha256'):
            assert len({s[key] for _, _, s in paired}) == 1
        assert len({tuple(s['augmentation_first_batch_sha256']) for _, _, s in paired}) == 1
        order = np.load(paired[0][1] / 'train_order.npy')
        conventions = set()
        for _, directory, _ in paired:
            np.testing.assert_array_equal(order, np.load(directory / 'train_order.npy'))
            with np.load(directory / 'mechanism_trace.npz') as trace:
                np.testing.assert_array_equal(expected_indices, trace['coordinate_indices'])
            cfg = yaml.safe_load((directory / 'config.yaml').read_text())
            conventions.add((cfg['augmentation_rng_convention'], cfg['num_workers']))
        assert len(conventions) == 1
        for noise in ('prefix', 'momentum'):
            directories = {j['method']: d for j, d, _ in paired}
            with np.load(directories[f'{noise}_standard'] / 'matrices.npz') as a, \
                 np.load(directories[f'{noise}_scale'] / 'matrices.npz') as b:
                for key in ('noising_coefficients', 'strategy', 'workload_coefficients', 'W'):
                    np.testing.assert_array_equal(a[key], b[key])
                # Clip changes the calibrated scalar, never the factorization.
                np.testing.assert_allclose(a['M'] / a['innovation_std_sum'],
                                           b['M'] / b['innovation_std_sum'], atol=1e-15)


def aggregate_final(results, configs, base):
    assert len(results) == 18
    verify_pairing(results)
    write_csv(base / 'final_multiseed.csv', result_rows(results))
    effects = []
    for seed in FINAL_SEEDS:
        accuracy = {j['method']: s['final']['test_top1'] for j, _, s in results if j['seed'] == seed}
        effects.append(dict(seed=seed, **factorial_effects(accuracy)))
    write_csv(base / 'factorial_effects.csv', effects)
    save_json(base / 'factorial_effects.json', dict(units='absolute top1 fraction', paired=effects,
        aggregate={key: mean_std([r[key] for r in effects]) for key in effects[0] if key != 'seed'}))
    final_summary, mechanism_summary, distortions = {}, {}, []
    for method in CELLS:
        trials = sorted((j['seed'], d, s) for j, d, s in results if j['method'] == method)
        assert [seed for seed, _, _ in trials] == list(FINAL_SEEDS)
        top1 = [s['final']['test_top1'] for _, _, s in trials]
        stats = mean_std(top1)
        final_summary[method] = dict(chosen_hyperparameters=configs[method], seeds=list(FINAL_SEEDS),
            top1_each_seed=top1, top1_mean=stats['mean'], top1_std=stats['sample_std'])
        per_trial = []
        per_distortion = []
        for seed, directory, _ in trials:
            with (directory / 'mechanism_metrics.csv').open() as f:
                metrics = list(csv.DictReader(f))
            per_trial.append({key: float(np.mean([float(row[key]) for row in metrics if row[key] != '']))
                              for key in ('r_cv', 'r_anisotropy', 'temporal_drift')})
            if CELLS[method]['noise'] != 'iid':
                with np.load(directory / 'mechanism_trace.npz') as trace, \
                     np.load(directory / 'matrices.npz') as matrices:
                    diagnostic = mf_distortion(trace['r_trace'], matrices['M'], matrices['W'])
                distortions.append(dict(method=method, seed=seed, **diagnostic))
                per_distortion.append(diagnostic)
        entry = dict(top1_mean=stats['mean'], top1_std=stats['sample_std'],
            **{key + '_mean': float(np.mean([r[key] for r in per_trial])) for key in per_trial[0]})
        if per_distortion:
            entry.update(mf_distortion_ratio_median=float(np.median([r['ratio_p50'] for r in per_distortion])),
                         mf_distortion_logabs_mean=float(np.mean([r['logabs_mean'] for r in per_distortion])))
        mechanism_summary[method] = entry
    write_csv(base / 'mf_distortion.csv', distortions)
    save_json(base / 'final_summary.json', final_summary)
    save_json(base / 'mechanism_summary.json', dict(cells=mechanism_summary,
        pairs={noise: [f'{noise}_standard', f'{noise}_scale'] for noise in ('iid', 'prefix', 'momentum')},
        diagnostic_scope='2048 fixed sampled coordinates; linearized multiplier, not exact Adam noise variance',
        aggregation='equal trial means of step metrics; drift excludes first step; MF ratio is median of trial medians',
        top1_units='fraction', top1_std='sample std across three seeds (ddof=1)'))


def main(args):
    assert torch.cuda.device_count() >= 4, 'GPUs 0,1,2,3 required'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '0,1,2,3')
    load_config(args.config)
    checkpoint_path()
    base = ROOT / 'exp2/results'
    (base / 'search').mkdir(parents=True, exist_ok=True)
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
            paired_initialization_order_and_coordinates=True, pipeline_stage='smoke only'))
        return 0
    first = run_queue(first_search_trials(), base / 'search', args.config)
    best_clips = {method: best([r for r in first if r[0]['method'] == method], 'max_grad_norm')[0]['max_grad_norm']
                  for method in ('iid_scale', 'momentum_scale')}
    second = run_queue(second_search_trials(best_clips), base / 'search', args.config)
    assert len(first) + len(second) == 23
    write_csv(base / 'search_summary.csv', result_rows(first + second))
    configs = select_configs(first, second)
    selected_path = base / 'selected_configs.json'
    if selected_path.exists():
        assert json.loads(selected_path.read_text()) == configs, 'Frozen configs differ from search'
    else:
        save_json(selected_path, configs)
    frozen = json.loads(selected_path.read_text())
    final = run_queue(final_trials(frozen), base / 'final', args.config)
    aggregate_final(final, frozen, base)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'exp2/config.yaml')
    parser.add_argument('--smoke', action='store_true')
    sys.exit(main(parser.parse_args()))
