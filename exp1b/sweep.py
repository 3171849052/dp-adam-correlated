"""Queue 5 single-GPU trials; summarize every trial without selecting winners."""
import argparse
import csv
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

from exp1b.train import METHODS, ROOT, load_config


def trials():
    rates = {'adam': ((3e-4, '3e-4'), (5e-4, '5e-4')),
             'dp_adam': ((5e-4, '5e-4'), (3e-3, '3e-3'), (5e-3, '5e-3'))}
    return [dict(method=method, lr=lr, max_grad_norm=None if method == 'adam' else 1.0,
                 name=f'lr_{label}')
            for method in METHODS for lr, label in rates[method]]


def write_summary(jobs, base, config, statuses):
    fields = ['method', 'lr', 'max_grad_norm', 'seed']
    fields += [f'test_top1_epoch_{e+1}' for e in range(config['epochs'])]
    fields += ['final_test_top1', 'train_loss', 'clip_fraction', 'epsilon', 'norm_p10', 'norm_p25', 'norm_p50', 'norm_p75', 'norm_p90', 'norm_p99', 'head_norm_median', 'backbone_norm_median', 'noise_std',
               'delta', 'target_mu', 'strategy_sensitivity', 'innovation_std_sum',
               'wall_seconds', 'status', 'exit_code', 'initialization_sha256', 'result_dir']
    with open(base.parent / 'sweep_summary.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for job in jobs:
            directory = base / job['method'] / job['name']
            row = {key: job[key] for key in fields[:3]}
            row.update(seed=config['seed'], status=statuses.get(str(directory), ('pending', None, None))[0],
                       exit_code=statuses.get(str(directory), ('pending', None, None))[1],
                       wall_seconds=statuses.get(str(directory), ('pending', None, None))[2],
                       result_dir=str(directory))
            if row['status'] == 'completed':
                summary = json.loads((directory / 'summary.json').read_text())
                privacy = summary['calibration']
                for epoch in summary['epochs']:
                    row[f"test_top1_epoch_{epoch['epoch']}"] = epoch['test_top1']
                row.update(final_test_top1=summary['final']['test_top1'],
                           train_loss=summary['final']['train_loss'],
                           clip_fraction=summary['final']['clip_fraction'],
                           wall_seconds=summary['wall_seconds'],
                           initialization_sha256=summary['initialization_sha256'])
                norms = summary['final_norm_stats']
                if norms:
                    row.update({f'norm_{q}': norms['full_model_norm'][q] for q in ('p10', 'p25', 'p50', 'p75', 'p90', 'p99')})
                    row.update(head_norm_median=norms['head_norm']['p50'], backbone_norm_median=norms['backbone_norm']['p50'])
                row['noise_std'] = summary['final']['noise_std']
                if privacy:
                    row.update(epsilon=config['privacy']['epsilon'],
                               delta=config['privacy']['delta'], target_mu=privacy['target_mu'],
                               strategy_sensitivity=privacy['sensitivity'],
                               innovation_std_sum=privacy['innovation_std_sum'])
            writer.writerow(row)


def main(args):
    assert torch.cuda.device_count() >= 4, 'GPUs 0,1,2,3 are required'
    config = load_config(args.config)
    jobs = trials()
    base = ROOT / 'exp1b/results/sweep'
    base.mkdir(parents=True, exist_ok=True)
    (ROOT / 'exp1b/cache/tmp').mkdir(parents=True, exist_ok=True)
    pending = deque(jobs)
    active, statuses = {}, {}
    write_summary(jobs, base, config, statuses)
    while pending or active:
        for gpu in range(4):
            if gpu in active:
                process, stream, directory, started = active[gpu]
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                statuses[str(directory)] = ('completed' if code == 0 else 'failed', code, time.monotonic() - started)
                del active[gpu]
                write_summary(jobs, base, config, statuses)
                print(f'{directory.relative_to(base)}: exit {code}', flush=True)
            if pending:
                job = pending.popleft()
                directory = base / job['method'] / job['name']
                directory.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, '-u', '-m', 'exp1b.train', '--config', str(args.config),
                       '--method', job['method'], '--lr', str(job['lr']), '--result-dir', str(directory)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                           PYTHONDONTWRITEBYTECODE='1',
                           TMPDIR=str(ROOT / 'exp1b/cache/tmp'))
                stream = open(directory / 'train.log', 'w')
                statuses[str(directory)] = ('running', None, None)
                write_summary(jobs, base, config, statuses)
                active[gpu] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT), stream, directory, time.monotonic())
                print(f'GPU {gpu}: {job["method"]}/{job["name"]}', flush=True)
        if active:
            time.sleep(0.1)
    return int(any(status != 'completed' for status, _, _ in statuses.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'exp1b/config.yaml')
    sys.exit(main(parser.parse_args()))
