"""Validated platform followed immediately by bounded adaptive search, seed 20261001."""
import argparse
import fcntl
import json
import math
import subprocess
import sys
from exp7 import BASE, ROOT
from exp7.config import IID, MOMENTUM, SCALE, METHODS, trial, trial_id, save_json
from exp7.launcher import run_queue
from exp7.report import best, export, rows

EPS_GRID = (1e-8, 1e-4, 1e-3, 1e-2, .1, .3, 1.)
LR_GRID = (.0005, .001, .002, .003, .005)
BUDGET = 28

def nearby(grid, value):
    index = min(range(len(grid)), key=lambda i: abs(math.log(grid[i] / value)))
    return grid[max(0, index-1):min(len(grid), index+2)]

def run_stage(name, jobs, gpus, reasons):
    known = {trial_id({k:r[k] for k in ('method', 'seed', 'lr', 'C', 'eps_scale')}) for r in rows()}
    unique = list({trial_id(j):j for j in jobs}.values())
    remaining = BUDGET - sum(r['source'] == 'new' for r in rows())
    accepted = []
    for j in unique:
        if trial_id(j) in known:
            accepted.append(j)
        elif remaining > 0:
            accepted.append(j)
            remaining -= 1
    save_json(BASE / f'results/{name}_candidates.json', unique)
    export(reasons, stage=name)
    result = run_queue(accepted, gpus)
    export(reasons, stage=name)
    return result

def at(method, **coordinates):
    return [r for r in rows() if r['method'] == method and all(
        r[k] == v if v is None else math.isclose(r[k], v, rel_tol=1e-10)
        for k, v in coordinates.items())]

def lr_bracket(candidates, winner):
    scored = [r for r in candidates if r['status'] == 'completed']
    return (any(r['lr'] < winner['lr'] and r['final_test_top1'] <= winner['final_test_top1'] for r in scored)
            and any(r['lr'] > winner['lr'] and r['final_test_top1'] <= winner['final_test_top1'] for r in scored))

def verify_platform(gpus):
    from exp7.history import import_history
    marker = BASE / 'results/platform_validation.json'
    # Reruns reuse completed smoke points; tests run in exp7 every invocation.
    with (BASE / 'results/unit_tests.log').open('w') as log:
        code = subprocess.call([sys.executable, '-B', '-m', 'pytest', 'exp7/tests', '-q',
                                '-o', 'cache_dir=exp7/runtime/pytest_cache',
                                '--basetemp=exp7/runtime/tests'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    assert code == 0, 'Unit tests failed; see exp7/results/unit_tests.log'
    history = import_history()
    jobs = [trial(IID, .0005, 30), trial(MOMENTUM, .005, 30), trial(SCALE, .002, 200, .1)]
    smokes = run_queue(jobs, gpus, smoke=True)
    reference = {m: best([r for r in history if r['method'] == m]) for m in METHODS}
    for row in smokes:
        assert row['status'] == 'completed' and row['smoke']
        assert row['optimizer_steps'] == row['noise_steps'] == 1 and row['physical_batches'] == 4
        assert row['initialization_sha256'] == reference[row['method']]['initialization_sha256']
        # Smoke hashes one logical batch; a historical epoch hashes fifty.
        assert row['augmentation_trace_sha256'][0] == smokes[0]['augmentation_trace_sha256'][0]
        import numpy as np
        np.testing.assert_array_equal(np.load(ROOT / row['result_dir'] / 'train_order.npy'),
                                      np.load(ROOT / reference[row['method']]['result_dir'] / 'train_order.npy'))
        assert row['calibration']['k'] == 5 and row['calibration']['b_participation'] == 50
    save_json(marker, dict(status='passed', unit_tests='passed', smoke_methods=list(METHODS),
              smoke_trials=3, gpus=gpus, history_audit='passed', smoke_excluded_from_search=True))

def search(gpus):
    reasons = {MOMENTUM:'frozen: historical C=10/30/100 and LR=.003/.005/.007 bracket; audited compatible'}
    # Coarse Scale points and missing IID LR points share the first FIFO batch.
    run_stage('coarse', [trial(IID, lr, 30) for lr in (.00025, .0005, .001)] +
              [trial(SCALE, .002, 20 / eps, eps) for eps in EPS_GRID], gpus, reasons)
    iid_candidates = at(IID, C=30)
    winner = best(iid_candidates)
    if not lr_bracket(iid_candidates, winner):
        lrs = [r['lr'] for r in iid_candidates]
        extra = winner['lr'] / 2 if winner['lr'] == min(lrs) else winner['lr'] * 2
        run_stage('iid_lr_extension', [trial(IID, extra, 30)], gpus, reasons)
        winner = best(at(IID, C=30))
    reasons[IID] = ('LR bracketed on both sides; historical C=10/30/100 bracket retained' if lr_bracket(at(IID, C=30), winner)
                    else 'one directional LR extension exhausted; C bracket retained')
    coarse = [r for r in at(SCALE, lr=.002) if math.isclose(r['C'] * r['eps_scale'], 20)]
    eps_best = best(coarse)['eps_scale']
    survivors = nearby(EPS_GRID, eps_best)
    run_stage('scale_joint', [trial(SCALE, .002, K / eps, eps) for eps in survivors for K in (10, 20, 40)], gpus, reasons)
    joint = best(at(SCALE, lr=.002))
    run_stage('scale_lr', [trial(SCALE, lr, joint['C'], joint['eps_scale']) for lr in LR_GRID], gpus, reasons)
    lr_candidates = at(SCALE, C=joint['C'], eps_scale=joint['eps_scale'])
    winner = best(lr_candidates)
    if not lr_bracket(lr_candidates, winner):
        lrs = [r['lr'] for r in lr_candidates]
        extra = winner['lr'] / 2 if winner['lr'] == min(lrs) else winner['lr'] * 2
        run_stage('scale_lr_extension', [trial(SCALE, extra, joint['C'], joint['eps_scale'])], gpus, reasons)
    winner = best(at(SCALE))
    # One small final round checks epsilon and K neighbors at the winning LR.
    # Include at most two epsilon and two K neighbors; no Cartesian refinement.
    eps, K, lr = winner['eps_scale'], winner['C'] * winner['eps_scale'], winner['lr']
    eps_neighbors = [e for e in nearby(EPS_GRID, eps) if e != eps]
    if eps == EPS_GRID[0]:
        eps_neighbors.append(eps / 10)
    elif eps == EPS_GRID[-1]:
        eps_neighbors.append(eps * 3)
    K_neighbors = [k for k in nearby((10, 20, 40), K) if not math.isclose(k, K)]
    if math.isclose(K, 10):
        K_neighbors.append(5)
    elif math.isclose(K, 40):
        K_neighbors.append(80)
    refinement = [trial(SCALE, lr, K / e, e) for e in eps_neighbors] + [trial(SCALE, lr, k / eps, eps) for k in K_neighbors]
    run_stage('scale_refinement', refinement, gpus, reasons)
    winner = best(at(SCALE))
    def coordinate_bracket(coordinate):
        eps, K = winner['eps_scale'], winner['C'] * winner['eps_scale']
        if coordinate == 'LR':
            return lr_bracket(at(SCALE, C=winner['C'], eps_scale=eps), winner)
        candidates = at(SCALE, lr=winner['lr'])
        if coordinate == 'epsilon':
            candidates = [r for r in candidates if math.isclose(r['C'] * r['eps_scale'], K)]
            value = lambda r:r['eps_scale']
        else:
            candidates = [r for r in candidates if r['eps_scale'] == eps]
            value = lambda r:r['C'] * r['eps_scale']
        center = value(winner)
        def worse(r):
            return r['status'] == 'numerical_failure' or r['final_test_top1'] <= winner['final_test_top1']
        return any(value(r) < center and worse(r) for r in candidates) and any(value(r) > center and worse(r) for r in candidates)
    brackets = {c:coordinate_bracket(c) for c in ('epsilon', 'K', 'LR')}
    reasons[SCALE] = ('epsilon/K/LR all bracketed at selected point' if all(brackets.values()) else
                     f'one bounded local refinement exhausted; bracket checks={brackets}; new-trial cap={BUDGET}')
    save_json(BASE / 'results/scale_brackets.json', brackets)
    export(reasons, complete=True, stage='done')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs=3, default=[0, 1, 2])
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    assert len(set(args.gpus)) == 3
    with (BASE / 'runtime/search.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_platform(args.gpus)
        if not args.smoke_only:
            state_path = BASE / 'results/search_state.json'
            if state_path.exists() and json.loads(state_path.read_text())['status'] == 'completed':
                print('Search already completed; reuse exp7/results/selected_configs.json', flush=True)
            else:
                search(args.gpus)

if __name__ == '__main__':
    main()
