"""Validate and execute the two-round search, with one CV worker per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from queue import Queue
from threading import Lock
import numpy as np
from exp11 import BASE, RESULTS, ROOT
from exp11.config import INITIAL_LAMBDAS, T, lambda_tag, trial_output, save_json, file_hash, code_hashes
from exp11.data import prepare
from exp11.matrices import get_matrix
from exp11.report import completed, rank, generate


def compatible_sources(recorded):
    current = code_hashes()
    if recorded == current:
        return True
    transition = RESULTS / 'gpu_transition.json'
    if transition.exists():
        value = json.loads(transition.read_text())
        return recorded == value['previous_code_sha256'] and current == value['current_code_sha256']
    return False


def run_worker(lam, smoke=False, gpu=0):
    output = trial_output(lam, smoke)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'summary.json'
    if summary_path.exists():
        r = json.loads(summary_path.read_text())
        if r['status'] != 'completed':
            raise RuntimeError(f'Previous failed trial requires inspection: {summary_path}')
        assert compatible_sources(r['code_sha256'])
        assert r['assets_manifest_sha256'] == file_hash(RESULTS / 'assets_manifest.json')
        assert r['optimizer_steps'] == (3 if smoke else T)
        assert r['official_evaluation_used'] is False and r['physical_gpu'] in (0, 1, 2, 3)
        for name, digest in r['file_sha256'].items():
            assert file_hash(output / name) == digest
        return r
    command = [sys.executable, '-B', '-m', 'exp11.train', '--lambda-value', repr(lam)]
    if smoke:
        command.append('--smoke')
    print(f'Start {"smoke" if smoke else "CV"} lambda={lam:.17g} GPU={gpu}', flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with (output / 'train.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    r = json.loads(summary_path.read_text())
    assert r['status'] == 'completed'
    if not smoke:
        previous = [p for p in completed() if p['lambda_value'] != lam]
        if previous:
            ref = previous[0]
            for key in ('initialization_sha256', 'classifier_initialization_sha256', 'train_order_sha256', 'rng_trace'):
                assert r[key] == ref[key], f'RNG pairing mismatch: {key}'
        print(f'Completed lambda={lam:.17g}, validation accuracy={r["accuracy"]:.6f}, loss={r["validation_loss"]:.6f}', flush=True)
    return r


def neighbors(lam):
    available = []
    for p in (RESULTS / 'matrices').glob('*/optimization.json'):
        meta = json.loads(p.read_text())
        value = meta['lambda_value']
        if value != lam:
            available.append((value, np.array(meta['coefficients'])))
    lower = [item for item in available if item[0] < lam]
    upper = [item for item in available if item[0] > lam]
    return ([max(lower, key=lambda item: item[0])] if lower else []) + ([min(upper, key=lambda item: item[0])] if upper else [])


def refinement_plan(rows):
    top = rank([r for r in rows if r['lambda_value'] > 0])[:2]
    existing = [r['lambda_value'] for r in rows]
    initial_logs = np.log10(INITIAL_LAMBDAS[1:])
    proposals = []
    for winner in top:
        center = np.log10(winner['lambda_value'])
        left = initial_logs[initial_logs < center - 1e-12]
        right = initial_logs[initial_logs > center + 1e-12]
        a = left[-1] if len(left) else center - .25
        b = right[0] if len(right) else center + .25
        for log_value in (center + (a-center)/3, center + 2*(a-center)/3,
                          center + (b-center)/3, center + 2*(b-center)/3):
            value = float(10. ** log_value)
            if any(np.isclose(value, x, rtol=1e-12, atol=0.) for x in existing + proposals):
                continue
            proposals.append(value)
    assert len(proposals) <= 8
    return dict(selection='Final internal validation accuracy descending, then loss, then lambda',
                top_two_positive=[dict(lambda_value=r['lambda_value'], accuracy=r['accuracy'],
                                       validation_loss=r['validation_loss']) for r in top],
                initial_completed=30, lambdas=sorted(proposals),
                rule='Two logarithmic trisection points on either side of each top positive lambda; '
                     'edge brackets extend by 0.25 decades; duplicates removed')


def run_round(lambdas, gpus, phase, total, adopted_pid=None):
    free = Queue()
    for gpu in gpus:
        if not (adopted_pid and gpu == 0):
            free.put(gpu)
    active = {}
    lock = Lock()

    def worker(lam):
        gpu = free.get()
        with lock:
            active[gpu] = lam
        try:
            return run_worker(lam, gpu=gpu)
        finally:
            with lock:
                del active[gpu]
            free.put(gpu)

    def adopted():
        with lock:
            active[0] = 0.
        path = trial_output(0.) / 'summary.json'
        while not path.exists():
            # Explicitly preserve the original worker during the four-GPU switch.
            os.kill(adopted_pid, 0)
            time.sleep(1)
        r = run_worker(0., gpu=0)
        with lock:
            del active[0]
        free.put(0)
        print('Preserved original lambda=0 worker completed', flush=True)
        return r

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        pending = set()
        if adopted_pid:
            assert 0 in gpus and phase == 'initial'
            pending.add(pool.submit(adopted))
        for lam in lambdas:
            if adopted_pid and lam == 0.:
                continue
            get_matrix(lam, neighbors(lam))
            pending.add(pool.submit(worker, lam))
        while pending:
            with lock:
                snapshot = [dict(physical_gpu=gpu, lambda_value=lam) for gpu, lam in sorted(active.items())]
            save_json(RESULTS / 'progress.json', dict(status='running', phase=phase,
                      completed_trainings=len(completed()), active=snapshot, planned_trainings=total,
                      gpus=gpus, pid=os.getpid()))
            done, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()  # Failures are surfaced; no failed trial is skipped or retried.
            if done:
                generate(total if phase == 'refinement' else None)


def execute(gpus, adopted_pid=None):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    started = time.time()
    prepare()
    tests = [sys.executable, '-B', '-m', 'pytest', 'exp11/tests', '-q', '-p', 'no:cacheprovider',
             '--basetemp=' + str(BASE / 'runtime/pytest_tmp'), '--junitxml=' + str(RESULTS / 'tests.xml')]
    with (RESULTS / 'tests.log').open('w') as log:
        subprocess.run(tests, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Mathematical and training-semantics tests passed', flush=True)
    smoke = run_worker(0., smoke=True, gpu=0)
    save_json(RESULTS / 'stage1.json', dict(status='passed', tests='tests.xml', smoke=smoke,
                                         code_sha256=code_hashes(), official_evaluation_used=False))
    save_json(RESULTS / 'initial_plan.json', dict(lambdas=INITIAL_LAMBDAS, seed=20261101,
              trainings=30, max_trainings=38, physical_gpus=gpus, max_workers_per_gpu=1,
              adopted_lambda_zero_pid=adopted_pid, code_sha256=code_hashes()))
    print('Tests and GPU smoke passed; executing all CV searches immediately', flush=True)
    run_round(INITIAL_LAMBDAS, gpus, 'initial', 38, adopted_pid)
    decision_path = RESULTS / 'refinement_plan.json'
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
    else:
        decision = refinement_plan(completed())
        save_json(decision_path, decision)
    total = 30 + len(decision['lambdas'])
    run_round(decision['lambdas'], gpus, 'refinement', total)
    rows = completed()
    assert len(rows) == total <= 38
    assert all(r['completed_epochs'] == 5 and r['optimizer_steps'] == 225 for r in rows)
    generate(total)
    save_json(RESULTS / 'progress.json', dict(status='completed', completed_trainings=total,
              planned_trainings=total, best_lambda=rank(rows)[0]['lambda_value'], seconds=time.time()-started))
    print(json.dumps(json.loads((RESULTS / 'best.json').read_text()), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--adopt-lambda-zero-pid', type=int)
    args = parser.parse_args()
    assert sorted(args.gpus) == [0, 1, 2, 3]
    try:
        execute(args.gpus, args.adopt_lambda_zero_pid)
    except Exception as error:
        save_json(RESULTS / 'failure.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
                  traceback=traceback.format_exc(), completed_trainings=len(completed()),
                  remaining_initial_lambdas=[v for v in INITIAL_LAMBDAS if v not in [r['lambda_value'] for r in completed()]]))
        raise
