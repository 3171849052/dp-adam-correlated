"""Seven trials on GPUs 0–3, one process per GPU, no winner selection."""
import argparse
from collections import deque
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
from datetime import datetime
import torch
from exp1d.train import ROOT, load_config


def trials():
    rates = ((2e-3, '2e-3'), (4e-3, '4e-3'), (5e-3, '5e-3'), (7e-3, '7e-3'))
    momentum = [dict(method='dp_adam_bandinvmf_momentum', lr=lr, eps_scale=None,
                     max_grad_norm=1., num_bands=4, name=f'lr_{label}') for lr, label in rates]
    scale = [dict(method='dp_adam_bandinvmf_scale', lr=1e-3, eps_scale=.1,
                  max_grad_norm=float(clip), num_bands=4,
                  name=f'lr_1e-3_eps_1e-1_clip_{clip}') for clip in (10, 100, 1000)]
    return momentum + scale


def write_summary(jobs, base, config, statuses):
    fields = ['method','lr','eps_scale','max_grad_norm','num_bands','seed']
    fields += [f'test_top1_epoch_{e+1}' for e in range(config['epochs'])]
    fields += ['final_test_top1','train_loss','clip_fraction','epsilon','delta','target_mu',
               'strategy_sensitivity','innovation_std_sum','final_marginal_noise_std',
               'scaled_norm_p50','scaled_norm_p90','scaled_norm_p99','unscaled_norm_p50',
               'sqrt_vhat_p50','sqrt_vhat_p90','scale_p50','wall_seconds','status','exit_code',
               'initialization_sha256','result_dir']
    with (base.parent / 'sweep_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            directory = base / job['method'] / job['name']
            status, code, seconds = statuses.get(str(directory), ('pending', None, None))
            row = {key: job[key] for key in fields[:5]}
            row.update(seed=config['seed'], epsilon=config['privacy']['epsilon'], delta=config['privacy']['delta'],
                       status=status, exit_code=code, wall_seconds=seconds, result_dir=str(directory))
            if status == 'completed':
                summary = json.loads((directory / 'summary.json').read_text())
                calibration = summary['calibration']
                for epoch in summary['epochs']:
                    row[f"test_top1_epoch_{epoch['epoch']}"] = epoch['test_top1']
                row.update(final_test_top1=summary['final']['test_top1'], train_loss=summary['final']['train_loss'],
                           clip_fraction=summary['final']['clip_fraction'], target_mu=calibration['target_mu'],
                           strategy_sensitivity=calibration['sensitivity'], innovation_std_sum=calibration['innovation_std_sum'],
                           final_marginal_noise_std=summary['final']['marginal_noise_std'],
                           wall_seconds=summary['wall_seconds'], initialization_sha256=summary['initialization_sha256'])
                if job['eps_scale'] is not None:
                    for group, qs in [('scaled_norm', ('p50','p90','p99')), ('unscaled_norm', ('p50',))]:
                        for q in qs:
                            row[f'{group}_{q}'] = summary['final_norm_stats'][group][q]
                    for group, qs in [('sqrt_vhat', ('p50','p90')), ('scale', ('p50',))]:
                        for q in qs:
                            row[f'{group}_{q}'] = summary['final_scale_stats'][group][q]
            writer.writerow(row)


def main(args):
    assert torch.cuda.device_count() >= 4, 'GPUs 0,1,2,3 are required'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '0,1,2,3'), 'Launcher requires physical GPUs 0,1,2,3'
    config = load_config(args.config)
    jobs = trials()
    base = ROOT / 'exp1d/results/sweep'
    if base.exists() or (base.parent / 'sweep_summary.csv').exists():
        runs = ROOT / 'exp1d/results/runs'
        runs.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=runs))
        base = run / 'sweep'
    base.mkdir(parents=True, exist_ok=True)
    print(f'Results: {base}', flush=True)
    print(f'Summary: {base.parent / "sweep_summary.csv"}', flush=True)
    (ROOT / 'exp1d/cache/tmp').mkdir(parents=True, exist_ok=True)
    pending, statuses = deque(), {}
    for job in jobs:
        directory = base / job['method'] / job['name']
        assert not directory.exists(), f'Refusing to overwrite existing trial: {directory}'
        pending.append(job)
    active = {}
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
                directory.parent.mkdir(parents=True, exist_ok=True)
                directory.mkdir()
                stream = (directory / 'train.log').open('x')
                cmd = [sys.executable, '-u', '-m', 'exp1d.train', '--config', str(args.config),
                       '--method', job['method'], '--lr', str(job['lr']), '--max-grad-norm', str(job['max_grad_norm']),
                       '--result-dir', str(directory)]
                if job['eps_scale'] is not None:
                    cmd += ['--eps-scale', str(job['eps_scale'])]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                           PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'exp1d/cache/tmp'))
                statuses[str(directory)] = ('running', None, None)
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[gpu] = (process, stream, directory, time.monotonic())
                write_summary(jobs, base, config, statuses)
                print(f'GPU {gpu}: {job["method"]}/{job["name"]}', flush=True)
        if active:
            time.sleep(.1)
    return int(any(status != 'completed' for status, _, _ in statuses.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'exp1d/config.yaml')
    sys.exit(main(parser.parse_args()))
