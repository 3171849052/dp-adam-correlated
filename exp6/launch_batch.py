"""FIFO on physical GPUs 1,2,3; exclusive per-GPU process locks."""
from exp6.runtime import ROOT, EXP, output_path, require_curve
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from exp6.config import Trial, METHODS, GPUS, SMOKE_STEPS


def read_completed(job):
    if not job.get('smoke',False):
        from exp6.audit import completed
        return completed(job)
    folder = output_path(job['result_dir'])
    summary = json.loads((folder / 'summary.json').read_text())
    config = json.loads((folder / 'config.json').read_text())
    trial = Trial(**job['trial'])
    assert summary['status'] == 'completed' and summary['smoke'] == job.get('smoke', False)
    assert summary['trial_id'] == config['trial_id'] == trial.identity
    assert summary['trial'] == config['trial'] == trial.asdict()
    assert summary['optimizer_steps'] == summary['noise_draws'] == (SMOKE_STEPS if summary['smoke'] else 250)
    if not summary['smoke']:
        assert abs(summary['final_privacy']['epsilon']-8) < 1e-7
    return summary


def launch(jobs):
    require_curve()
    assert len({str(output_path(j['result_dir'])) for j in jobs}) == len(jobs)
    queue, reused, trained = [], [], []
    for job in jobs:
        if (output_path(job['result_dir']) / 'summary.json').exists():
            read_completed(job)
            reused.append(Trial(**job['trial']).identity)
        else:
            folder = output_path(job['result_dir'])
            assert not folder.exists() or not any(folder.iterdir()), f'Incomplete trial: {folder}'
            queue.append(job)
    locks, active = [], {}
    try:
        for gpu in GPUS:
            handle = (EXP / 'runtime' / f'gpu_{gpu}.lock').open('a')
            locks.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while queue or active:
            for gpu in GPUS:
                if gpu in active or not queue:
                    continue
                job = queue.pop(0)
                trial = Trial(**job['trial'])
                folder = output_path(job['result_dir'])
                folder.mkdir(parents=True, exist_ok=True)
                assert not any(p.name != 'train.log' for p in folder.iterdir()), f'Incomplete trial: {folder}'
                log = (folder / 'train.log').open('w')
                cmd = [sys.executable, '-B', '-m', 'exp6.train', '--method', trial.method,
                    '--seed', str(trial.seed), '--lr', str(trial.lr), '--C', str(trial.C),
                    '--geom-eps', str(trial.geom_eps), '--physical-batch-size', str(trial.physical_batch_size),
                    '--result-dir', str(folder)]
                if job.get('smoke', False):
                    cmd.append('--smoke')
                process = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                           stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = process, log, job
                print(json.dumps(dict(event='started', gpu=gpu, pid=process.pid, method=trial.method)), flush=True)
            for gpu, (process, log, job) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                assert code == 0, f'Trial exited {code}: {job["result_dir"]}/train.log'
                read_completed(job)
                trained.append(Trial(**job['trial']).identity)
                print(json.dumps(dict(event='completed', gpu=gpu, method=job['trial']['method'])), flush=True)
            if active:
                time.sleep(1)
    finally:
        for process, log, _ in active.values():
            process.terminate()
            process.wait()
            log.close()
        for handle in locks:
            handle.close()

    return dict(trained=trained,reused=reused)


def smoke_jobs():
    return [dict(trial=Trial(method).asdict(), smoke=True,
                 result_dir=str(EXP / 'results/smoke' / method)) for method in METHODS]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', choices=['1,2,3'], default='1,2,3')
    p.add_argument('--smoke', action='store_true', required=True)
    p.parse_args()
    launch(smoke_jobs())


if __name__ == '__main__':
    main()
