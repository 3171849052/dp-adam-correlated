"""Three physical GPUs, global FIFO, one subprocess per GPU."""
from collections import deque
import json
import os
import subprocess
import sys
import time
from exp7b import BASE, ROOT
from exp7b.config import save_json, trial_id, trial_dir, SEED


def run_queue(jobs, gpus, category='search', command=None, poll_seconds=1):
    assert sorted(gpus) == [1, 2, 3], 'Use physical GPUs 1, 2, 3; GPU 0 is reserved'
    assert category in ('smoke', 'search', 'final')
    jobs = list({trial_id(j): j for j in jobs}.values())
    pending, active, results = deque(), {}, []
    for job in jobs:
        assert category == 'smoke' or (job['seed'] == SEED) == (category == 'search')
        directory = trial_dir(job, category)
        if directory.exists():
            path = directory / 'summary.json'
            assert path.exists(), f'Inspect incomplete trial: {directory}'
            row = json.loads(path.read_text())
            assert all(row[k] == v for k, v in job.items())
            assert row['smoke'] == (category == 'smoke')
            assert row['status'] in ('completed', 'numerical_failure'), row
            if row['status'] == 'completed' and command is None:
                from exp7b.audit import audit_trial
                audit_trial(directory)
            results.append(row)
        else:
            pending.append((job, directory))
    events = BASE / 'results/scheduler.jsonl'
    events.parent.mkdir(parents=True, exist_ok=True)
    def event(kind, gpu, job, **extra):
        row = dict(event=kind, time=time.time(), gpu=gpu, trial_id=trial_id(job), category=category, **job, **extra)
        with events.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    try:
        while pending or active:
            for gpu in gpus:
                if gpu not in active:
                    continue
                process, job, directory = active[gpu]
                code = process.poll()
                if code is None:
                    continue
                path = directory / 'summary.json'
                if not path.exists():
                    save_json(path, dict(job, status='failed', smoke=category == 'smoke', source='new',
                              final_test_top1=None, error=f'Process exited {code} without summary',
                              result_dir=str(directory.relative_to(ROOT))))
                row = json.loads(path.read_text())
                event('finish', gpu, job, returncode=code, status=row['status'])
                results.append(row)
                del active[gpu]
                if row['status'] == 'failed':
                    raise RuntimeError(row)
                assert (code, row['status']) in ((0, 'completed'), (2, 'numerical_failure'))
                if row['status'] == 'completed' and command is None:
                    from exp7b.audit import audit_trial
                    audit_trial(directory)
            # Rotate first choice between queue invocations to spread algorithms over GPUs.
            starts = sum(json.loads(s)['event'] == 'start' for s in events.read_text().splitlines()) if events.exists() else 0
            offset = starts % 3
            for gpu in gpus[offset:] + gpus[:offset]:
                if gpu in active or not pending:
                    continue
                job, directory = pending.popleft()
                if command:
                    args = command(job, directory, gpu)
                else:
                    args = [sys.executable, '-B', '-m', 'exp7b.train', '--method', job['method'],
                            '--seed', str(job['seed']), '--lr', str(job['lr']), '--max-grad-norm', str(job['C']),
                            '--result-dir', str(directory)]
                    if job['eps_scale'] is not None:
                        args += ['--eps-scale', str(job['eps_scale'])]
                    if category == 'smoke':
                        args += ['--smoke']
                process = subprocess.Popen(args, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
                active[gpu] = process, job, directory
                event('start', gpu, job, pid=process.pid)
            if active:
                time.sleep(poll_seconds)
    finally:
        for gpu, (process, job, directory) in active.items():
            process.terminate()
            process.wait()
            event('finish', gpu, job, returncode=process.returncode, status='interrupted')
    return results
