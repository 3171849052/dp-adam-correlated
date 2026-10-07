"""FIFO queue: exactly three distinct GPUs, at most one live trial per GPU."""
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from exp7 import BASE, ROOT
from exp7.config import save_json, trial_id

def run_queue(jobs, gpus, smoke=False, command=None, poll_seconds=2):
    assert len(gpus) == 3 and len(set(gpus)) == 3
    jobs = list({trial_id(j): j for j in jobs}.values())
    pending, active, results = deque(), {}, []
    category = 'smoke' if smoke else 'trials'
    events = BASE / 'results/scheduler.jsonl'
    for job in jobs:
        directory = BASE / 'results' / category / trial_id(job)
        if directory.exists():
            path = directory / 'summary.json'
            if not path.exists():
                raise RuntimeError(f'Incomplete trial requires manual cleanup: {directory}')
            summary = json.loads(path.read_text())
            assert all(summary[k] == v for k, v in job.items()) and summary['smoke'] == smoke
            if summary['status'] in ('completed', 'numerical_failure'):
                results.append(summary)
                continue
            raise RuntimeError(f'Failed trial requires manual cleanup: {directory}')
        pending.append((job, directory))
    def event(kind, gpu, job, **extra):
        row = dict(event=kind, time=time.time(), gpu=gpu, trial_id=trial_id(job), **job, **extra)
        with events.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    while pending or active:
        for gpu in gpus:
            if gpu in active:
                process, job, directory = active[gpu]
                code = process.poll()
                if code is None:
                    continue
                path = directory / 'summary.json'
                if not path.exists():
                    directory.mkdir(parents=True, exist_ok=True)
                    save_json(path, dict(job, status='failed', smoke=smoke, source='new',
                                         final_test_top1=None, error=f'Process exited {code} without summary',
                                         result_dir=str(directory.relative_to(ROOT))))
                summary = json.loads(path.read_text())
                event('finish', gpu, job, returncode=code, status=summary['status'])
                results.append(summary)
                del active[gpu]
            if pending and gpu not in active:
                job, directory = pending.popleft()
                args = (command(job, directory, gpu) if command else
                        [sys.executable, '-B', '-m', 'exp7.train', '--method', job['method'],
                         '--lr', str(job['lr']), '--max-grad-norm', str(job['C']),
                         '--result-dir', str(directory)])
                if command is None:
                    if job['eps_scale'] is not None:
                        args += ['--eps-scale', str(job['eps_scale'])]
                    if smoke:
                        args += ['--smoke']
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                process = subprocess.Popen(args, cwd=ROOT, env=env)
                active[gpu] = process, job, directory
                event('start', gpu, job, pid=process.pid, smoke=smoke)
        if active:
            time.sleep(poll_seconds)
    failures = [r for r in results if r['status'] == 'failed']
    if failures:
        raise RuntimeError(f"Training failed: {[r['result_dir'] for r in failures]}")
    return results
