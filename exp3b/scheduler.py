"""Offline subprocess batches: at most one job on each of GPUs 0,1,2,3."""
import os
import subprocess
import sys
from exp3b import BASE, ROOT
from exp3b.spec import RunSpec, read_completed, write_json


def run_commands(commands, gpus, label):
    assert tuple(gpus) == (0, 1, 2, 3)
    logs = BASE / 'results/launch_logs'
    logs.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(commands), 4):
        running = []
        handles = []
        try:
            for offset, command in enumerate(commands[start:start + 4]):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[offset]),
                           PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
                log = (logs / f'{label}_{start + offset:03d}.log').open('w')
                handles.append(log)
                print(f'Launching {label} job {start + offset + 1}/{len(commands)} on GPU {gpus[offset]}', flush=True)
                running.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
            # A failed job stops its batch immediately, including siblings.
            pending = set(running)
            while pending:
                for process in tuple(pending):
                    status = process.poll()
                    if status is not None:
                        assert status == 0, f'{label} subprocess failed with exit {status}; see {logs}'
                        pending.remove(process)
                if pending:
                    import time
                    time.sleep(.25)
        finally:
            for process in running:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
            for log in handles:
                log.close()


def run_trials(specs, gpus, label):
    commands = []
    for spec in specs:
        assert isinstance(spec, RunSpec)
        if spec.directory.exists():
            read_completed(spec)
            continue
        file = BASE / 'specs/generated' / f'{spec.directory.name}_{spec.fingerprint()[:12]}.json'
        write_json(file, spec.mapping())
        commands.append([sys.executable, '-m', 'exp3b.train', '--spec', str(file)])
    run_commands(commands, gpus, label)
    return [read_completed(spec) for spec in specs]
