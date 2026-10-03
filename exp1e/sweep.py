"""Four-GPU staged clip/LR search and independent three-seed validation."""
import argparse
from collections import deque
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np
import torch
from exp1e.train import ROOT, METHODS, load_config

SEARCH_SEED = 20261001
FINAL_SEEDS = (20261011, 20261012, 20261013)
LR_GRID = (5e-4, 1e-3, 2e-3, 3e-3)
SCALE = 'dp_adam_bandinvmf_scale'


def scale_job(clip, lr=1e-3):
    return dict(method=SCALE, lr=lr, eps_scale=.1, max_grad_norm=float(clip), num_bands=4, seed=SEARCH_SEED)


def stage1_trials():
    return [dict(scale_job(c), name=f'clip_{c}') for c in (100, 150, 200, 300, 500)]


def stage2_trials(clip):
    return [dict(scale_job(clip, lr), name=f'lr_{lr:g}') for lr in LR_GRID]


def selected_configs(clip, lr):
    return {'adam': dict(lr=3e-4), 'dp_adam': dict(lr=5e-4, max_grad_norm=1.),
            'dp_adam_bandinvmf_momentum': dict(lr=5e-3, max_grad_norm=1., num_bands=4),
            SCALE: dict(lr=lr, max_grad_norm=clip, eps_scale=.1, num_bands=4)}


def final_trials(configs):
    return [dict(configs[method], method=method, seed=seed, name=f'{method}/seed_{seed}')
            for seed in FINAL_SEEDS for method in METHODS]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_completed(directory, job):
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['status'] == 'completed' and not summary['smoke']
    for key in ('method', 'seed', 'lr', 'max_grad_norm', 'eps_scale'):
        assert summary[key] == job.get(key), (directory, key)
    for filename in ('config.yaml','train.log','metrics.csv','train_order.npy','final.pt'):
        assert (directory / filename).is_file(), (directory, filename)
    assert summary['optimizer_steps'] == summary['planned_total_steps'] == 250
    if job['method'] == SCALE:
        assert (directory / 'norm_stats.csv').is_file() and (directory / 'scale_stats.csv').is_file()
    return summary


def run_queue(jobs, base, config_path, smoke=False):
    """Refill an available GPU immediately; failed stages never select a winner."""
    pending, active, results = deque(), {}, []
    for job in jobs:
        directory = base / job['name']
        if directory.exists():
            assert not smoke
            results.append((job, directory, read_completed(directory, job)))
        else:
            pending.append(job)
    failed = False
    while pending or active:
        for gpu in range(4):
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
                    summary = (json.loads((directory/'summary.json').read_text()) if smoke
                               else read_completed(directory, job))
                    results.append((job, directory, summary))
            if pending:
                job = pending.popleft()
                directory = base / job['name']
                directory.mkdir(parents=True)
                stream = (directory/'train.log').open('x')
                cmd = [sys.executable, '-u', '-m', 'exp1e.train', '--config', str(config_path),
                       '--method', job['method'], '--seed', str(job['seed']), '--lr', str(job['lr']),
                       '--result-dir', str(directory)]
                for key, flag in [('max_grad_norm','--max-grad-norm'), ('eps_scale','--eps-scale')]:
                    if key in job:
                        cmd += [flag, str(job[key])]
                if smoke:
                    cmd += ['--smoke']
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                           TMPDIR=str(ROOT/'exp1e/cache/tmp'))
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[gpu] = (process, stream, job, directory)
                print(f'GPU {gpu}: {directory}', flush=True)
        if active:
            time.sleep(.1)
    if failed:
        raise RuntimeError('Trial failed; see its train.log. Pipeline stopped before next stage.')
    return results


def write_csv(path, results, stage):
    rows = []
    for job, directory, summary in results:
        rows.append(dict(stage=stage, **{k: job.get(k) for k in
                         ('method','seed','lr','eps_scale','max_grad_norm','num_bands')},
                         final_test_top1=summary['final']['test_top1'],
                         train_loss=summary['final']['train_loss'],
                         initialization_sha256=summary['initialization_sha256'], result_dir=str(directory)))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def best(results, key):
    return min(results, key=lambda item: (-item[2]['final']['test_top1'], item[0][key]))


def aggregate_final(results, configs, base):
    assert len(results) == 12
    for seed in FINAL_SEEDS:
        same_seed = [(d,s) for j,d,s in results if j['seed'] == seed]
        assert len(same_seed) == 4 and len({s['initialization_sha256'] for d,s in same_seed}) == 1
        order = np.load(same_seed[0][0]/'train_order.npy')
        for directory, _ in same_seed[1:]:
            np.testing.assert_array_equal(order, np.load(directory/'train_order.npy'))
    write_csv(base/'final_multiseed.csv', results, 'final')
    summaries = []
    for method in METHODS:
        records = sorted((j['seed'], s['final']) for j,d,s in results if j['method'] == method)
        seeds = [seed for seed,_ in records]
        assert seeds == list(FINAL_SEEDS)
        top1 = [r['test_top1'] for _,r in records]
        loss = [r['train_loss'] for _,r in records]
        summaries.append(dict(method=method, chosen_hyperparameters=configs[method], seeds=seeds,
                              test_top1_each_seed=top1, test_top1_mean=statistics.mean(top1),
                              test_top1_std=statistics.stdev(top1), train_loss_each_seed=loss,
                              train_loss_mean=statistics.mean(loss), train_loss_std=statistics.stdev(loss),
                              epsilon=None if method == 'adam' else 8.,
                              delta=None if method == 'adam' else 1e-5,
                              num_bands=configs[method].get('num_bands')))
    save_json(base/'final_summary.json', summaries)


def main(args):
    assert torch.cuda.device_count() >= 4, 'GPUs 0,1,2,3 required'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '0,1,2,3')
    load_config(args.config)
    base = ROOT/'exp1e/results'
    (base/'search').mkdir(parents=True, exist_ok=True)
    (ROOT/'exp1e/cache/tmp').mkdir(parents=True, exist_ok=True)
    if args.smoke:
        jobs = [j for j in final_trials(selected_configs(100.,1e-3)) if j['seed']==FINAL_SEEDS[0]]
        run_queue(jobs, base/'smoke', args.config, smoke=True)
        return 0
    first = run_queue(stage1_trials(), base/'search/stage1', args.config)
    first_rows = write_csv(base/'search/stage1_summary.csv', first, 'stage1')
    winner = best(first, 'max_grad_norm')
    clip = winner[0]['max_grad_norm']
    save_json(base/'selected_clip.json', dict(max_grad_norm=clip, selection_metric='final_test_top1',
                                           result_dir=str(winner[1])))
    # The completed stage-1 winner is the lr=1e-3 stage-2 trial.
    second = run_queue([j for j in stage2_trials(clip) if j['lr'] != 1e-3],
                       base/'search/stage2', args.config)
    second.append(winner)
    second_rows = write_csv(base/'search/stage2_summary.csv', second, 'stage2')
    with (base/'search_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(first_rows[0]))
        writer.writeheader(); writer.writerows(first_rows+second_rows)
    configs = selected_configs(clip, best(second,'lr')[0]['lr'])
    save_json(base/'selected_configs.json', configs)
    configs = json.loads((base/'selected_configs.json').read_text())
    final = run_queue(final_trials(configs), base/'final', args.config)
    aggregate_final(final, configs, base)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'exp1e/config.yaml')
    parser.add_argument('--smoke', action='store_true')
    sys.exit(main(parser.parse_args()))
