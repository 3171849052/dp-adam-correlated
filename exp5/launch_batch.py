"""FIFO single-GPU trials on physical GPUs 0,1,2; completed trials reused."""
from exp5.runtime import ROOT, EXP, output_path
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from exp5.config import FIXED, GPUS, Trial


def read_completed(trial, folder):
    folder = output_path(folder)
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['status'] == 'completed' and not summary['smoke']
    assert summary['trial_id'] == trial.identity and summary['fixed'] == FIXED
    assert all(summary[k] == v for k, v in trial.asdict().items())
    assert summary['optimizer_steps'] == summary['noise_steps'] == 250
    assert summary['physical_batches'] == 2500
    assert [r['logical_steps'] for r in summary['epochs']] == [50, 100, 150, 200, 250]
    assert abs(summary['final']['epsilon'] - 8.) < 1e-7
    return summary


def launch(jobs, on_complete=None):
    assert Path(sys.prefix).name == 'curve', 'Run in conda environment curve'
    assert len({str(output_path(j['result_dir'])) for j in jobs}) == len(jobs)
    queue, active = [], {}
    for job in jobs:
        folder = output_path(job['result_dir'])
        trial = Trial(**job['trial'])
        if (folder / 'summary.json').exists() and not job.get('smoke', False):
            read_completed(trial, folder)
            if on_complete:
                on_complete(job)
        else:
            if folder.exists():
                raise RuntimeError(f'Incomplete trial directory: {folder}; inspect it before restarting')
            queue.append(job)
    try:
        while queue or active:
            for gpu in GPUS:
                if gpu not in active and queue:
                    job = queue.pop(0)
                    trial = Trial(**job['trial'])
                    folder = output_path(job['result_dir'])
                    folder.mkdir(parents=True)
                    log = (folder / 'train.log').open('w')
                    command = [sys.executable, '-B', '-m', 'exp5.train', '--method', trial.method,
                               '--seed', str(trial.seed), '--lr', str(trial.lr), '--C', str(trial.C),
                               '--result-dir', str(folder)]
                    if job.get('smoke', False):
                        command.append('--smoke')
                    process = subprocess.Popen(command, cwd=ROOT,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                       PYTHONDONTWRITEBYTECODE='1'), stdout=log, stderr=subprocess.STDOUT)
                    active[gpu] = process, log, job
                    print(json.dumps(dict(event='started', gpu=gpu, pid=process.pid,
                                          trial_id=trial.identity, result_dir=str(folder))), flush=True)
            for gpu, (process, log, job) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del active[gpu]
                    if code:
                        raise RuntimeError(f'Trial exited {code}: {Path(job["result_dir"]) / "train.log"}')
                    summary = json.loads((Path(job['result_dir']) / 'summary.json').read_text())
                    assert summary['status'] == 'completed'
                    if on_complete:
                        on_complete(job)
                    print(json.dumps(dict(event='completed', gpu=gpu, result_dir=job['result_dir'])), flush=True)
            if active:
                time.sleep(1)
    finally:
        for process, log, _ in active.values():
            process.terminate()
            process.wait()
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--gpus', default='0,1,2', choices=['0,1,2'])
    a = p.parse_args()
    if a.smoke:
        from exp5.config import METHODS, SEARCH_SEED
        # Three simultaneous real trials also exercise GPU 2 independently.
        trials = [Trial(METHODS[0], SEARCH_SEED, 5e-4, 1.),
                  Trial(METHODS[1], SEARCH_SEED, 5e-4, 1.),
                  Trial(METHODS[0], SEARCH_SEED + 1, 5e-4, 1.)]
        launch([dict(trial=t.asdict(), result_dir=str(EXP / 'results/smoke' / t.identity),
                     smoke=True) for t in trials])
    else:
        from exp5.final_runner import run
        run()


if __name__ == '__main__':
    main()
