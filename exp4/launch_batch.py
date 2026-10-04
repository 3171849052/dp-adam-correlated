"""FIFO queue on physical GPUs 0,2,3; one process per GPU, at most 3 trials."""
from exp4.runtime import ROOT, EXP, output_path
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from exp4.config import METHODS


def trial_command(trial):
    assert trial['method'] in METHODS
    assert set(trial) <= {'method', 'seed', 'lr', 'update_clip_norm', 'result_dir', 'smoke', 'scale_probe'}
    assert trial['lr'] > 0 and trial['update_clip_norm'] > 0
    destination = output_path(trial['result_dir'])
    command = [sys.executable, '-B', '-m', 'exp4.train', '--method', trial['method'],
               '--seed', str(trial['seed']), '--lr', str(trial['lr']),
               '--update-clip-norm', str(trial['update_clip_norm']), '--result-dir', str(destination)]
    if trial.get('smoke', False):
        command.append('--smoke')
    if trial.get('scale_probe', False):
        command.append('--scale-probe')
    return command, destination


def launch(trials, on_complete=None):
    assert Path(sys.prefix).name == 'curve', 'Run in conda environment curve'
    assert len({str(output_path(t['result_dir'])) for t in trials}) == len(trials)
    queue = list(trials)
    active = {}
    trial_by_path = {str(output_path(t['result_dir'])): t for t in trials}
    try:
        while queue or active:
            for gpu in (0, 2, 3):
                if gpu not in active and queue:
                    trial = queue.pop(0)
                    command, destination = trial_command(trial)
                    destination.mkdir(parents=True, exist_ok=False)
                    log = (destination / 'train.log').open('w')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1')
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[gpu] = process, log, destination
                    print(json.dumps(dict(event='started', gpu=gpu, pid=process.pid,
                                          result_dir=str(destination))), flush=True)
            for gpu, (process, log, destination) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del active[gpu]
                    if code != 0:
                        raise RuntimeError(f'Trial exited {code}: {destination / "train.log"}')
                    assert json.loads((destination / 'summary.json').read_text())['status'] == 'completed'
                    if on_complete is not None:
                        on_complete(trial_by_path[str(destination)])
                    print(json.dumps(dict(event='completed', gpu=gpu, result_dir=str(destination))), flush=True)
            if active:
                time.sleep(1)
    finally:
        for process, log, _ in active.values():
            process.terminate()
            process.wait()
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    launch(json.loads(args.spec.read_text()))


if __name__ == '__main__':
    main()
