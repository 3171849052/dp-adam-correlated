"""Queue 36 single-GPU trials; summarize every trial without selecting winners."""
import argparse
import csv
from collections import deque
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from exp1.train import METHODS, ROOT, load_config


def trials():
    jobs = []
    rates = ((1e-5, '1e-5'), (3e-5, '3e-5'), (1e-4, '1e-4'))
    for method in METHODS:
        for lr, label in rates:
            values = itertools.product(((1e-3, '1e-3'), (1e-2, '1e-2'), (1e-1, '1e-1')),
                                       (0.3, 1.0, 3.0)) if method.endswith('_scale') else [(None, 1.0)]
            for eps, clip in values:
                name = f'lr_{label}'
                if eps is not None:
                    name += f'_eps_{eps[1]}_clip_{clip:g}'
                jobs.append(dict(method=method, lr=lr, eps_scale=eps[0] if eps else None,
                                 max_grad_norm=clip if method != 'adam' else None,
                                 name=name))
    return jobs


def write_summary(jobs, base, config, statuses):
    fields = ['method', 'lr', 'eps_scale', 'max_grad_norm', 'seed']
    fields += [f'test_top1_epoch_{e+1}' for e in range(config['epochs'])]
    fields += ['final_test_top1', 'train_loss', 'clip_fraction', 'target_epsilon',
               'delta', 'target_mu', 'strategy_sensitivity', 'innovation_std_sum',
               'wall_seconds', 'status', 'exit_code', 'initialization_sha256', 'result_dir']
    with open(base.parent / 'sweep_summary.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for job in jobs:
            directory = base / job['method'] / job['name']
            row = {key: job[key] for key in fields[:4]}
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
                if privacy:
                    row.update(target_epsilon=config['privacy']['epsilon'],
                               delta=config['privacy']['delta'], target_mu=privacy['target_mu'],
                               strategy_sensitivity=privacy['sensitivity'],
                               innovation_std_sum=privacy['innovation_std_sum'])
            writer.writerow(row)


def main(args):
    config = load_config(args.config)
    jobs = trials()
    base = ROOT / 'exp1/results/sweep'
    base.mkdir(parents=True, exist_ok=True)
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
                cmd = [sys.executable, '-u', '-m', 'exp1.train', '--config', str(args.config),
                       '--method', job['method'], '--lr', str(job['lr']), '--result-dir', str(directory)]
                if job['max_grad_norm'] is not None:
                    cmd += ['--max-grad-norm', str(job['max_grad_norm'])]
                if job['eps_scale'] is not None:
                    cmd += ['--eps-scale', str(job['eps_scale'])]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                stream = open(directory / 'train.log', 'w')
                active[gpu] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT), stream, directory, time.monotonic())
                print(f'GPU {gpu}: {job["method"]}/{job["name"]}', flush=True)
        if active:
            time.sleep(1)
    return int(any(status != 'completed' for status, _, _ in statuses.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'exp1/config.yaml')
    sys.exit(main(parser.parse_args()))
