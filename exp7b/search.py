"""Bounded adaptive coordinate search; search seed only, accuracy only."""
import argparse
import json
import math
from exp7b import BASE
from exp7b.config import *
from exp7b.launcher import run_queue
from exp7b.history import reuse
from exp7b.report import rows, best, search_report
from exp7b.frozen import freeze

EPS = (.03, .1, .3, 1.)

def at(method, **coordinates):
    return [r for r in rows('search') if r['method'] == method and all(
        r[k] == v if v is None else r[k] is not None and math.isclose(r[k], v, rel_tol=1e-10)
        for k,v in coordinates.items())]

def run_stage(name, jobs, gpus):
    jobs = list({trial_id(j): j for j in jobs}.values())
    assert all(j['seed'] == SEED and j['method'] in SEARCH_METHODS for j in jobs)
    save_json(BASE / f'results/search/{name}_candidates.json', jobs)
    for j in jobs:
        reuse(j)
    result = run_queue(jobs, gpus, category='search')
    search_report()
    return result

def neighbors(grid, winner):
    i = grid.index(winner)
    return grid[max(0,i-1): min(len(grid),i+2)]

def extension(value, grid):
    if math.isclose(value, min(grid)):
        return value / 2
    if math.isclose(value, max(grid)):
        return value * 2
    return None

def lr_search(method, center, grid, gpus):
    run_stage(method + '_lr', [trial(method, lr, center['C'], center['eps_scale']) for lr in grid], gpus)
    winner = best(at(method, C=center['C'], eps_scale=center['eps_scale']))
    extra = extension(winner['lr'], grid)
    if extra is not None:
        run_stage(method + '_lr_extension', [trial(method, extra, center['C'], center['eps_scale'])], gpus)

def search(gpus):
    assert sorted(gpus) == [1,2,3]
    selected_path = BASE / 'results/search/selected_configs.json'
    if selected_path.exists():
        return freeze(json.loads(selected_path.read_text()))
    marker = json.loads((BASE / 'results/platform_validation.json').read_text())
    assert marker['status'] == 'passed' and marker['smoke_trials'] == 7
    # Independent first-round jobs share a global FIFO across all four methods.
    first = [trial(SGD, lr, 10) for lr in (.001,.002,.003)]
    first += [trial(MOMENTUM_SCALE,.005,10/eps,eps) for eps in EPS]
    first += [trial(BIAS,.005,C) for C in (3,10,30,100)]
    first += [trial(BIAS_SCALE,.005,10/eps,eps) for eps in EPS]
    run_stage('initial', first, gpus)
    reasons = {}

    w = best(at(SGD, C=10))
    extra = extension(w['lr'], (.001,.002,.003))
    if extra is not None:
        run_stage('sgd_lr_extension', [trial(SGD,extra,10)], gpus)
    reasons[SGD] = 'LR .001/.002/.003 at C=10; stop at internal winner or after one directional extension'

    # Verify the supplied bracket with compatible recorded evidence.
    bracket_jobs = [trial(MOMENTUM_SCALE,lr,100,.1) for lr in (.003,.005,.007)]
    bracket_jobs += [trial(MOMENTUM_SCALE,.005,C,.1) for C in (50,100,200)]
    run_stage('momentum_scale_existing_bracket', bracket_jobs, gpus)
    coarse = [r for r in at(MOMENTUM_SCALE, lr=.005) if math.isclose(r['C']*r['eps_scale'],10)]
    w = best(coarse)
    anchor = best(at(MOMENTUM_SCALE,lr=.005,C=100,eps_scale=.1))
    bracket = all(r['status'] != 'completed' or r['final_test_top1'] <= anchor['final_test_top1']
                  for r in at(MOMENTUM_SCALE,eps_scale=.1))
    if w['eps_scale'] == .1 and bracket:
        reasons[MOMENTUM_SCALE] = 'eps_scale=.1 wins K=10; compatible C/LR bracket retained'
    else:
        for_scale(MOMENTUM_SCALE, w, gpus, lr_grid=(.003,.005,.007), refine=False)
        reasons[MOMENTUM_SCALE] = 'eps neighborhood, K=5/10/20, then LR=.003/.005/.007 with at most one extension'

    w = best(at(BIAS,lr=.005))
    # A single C extension only when the best C is an outer endpoint.
    if w['C'] in (3.,100.):
        run_stage('bias_C_extension', [trial(BIAS,.005,1 if w['C'] == 3 else 300)], gpus)
        w = best(at(BIAS,lr=.005))
    lr_search(BIAS, w, (.002,.003,.005,.007,.01), gpus)
    reasons[BIAS] = 'C search at LR=.005, then LR search at best C; at most one endpoint extension per coordinate'

    w = best(at(BIAS_SCALE,lr=.005))
    for_scale(BIAS_SCALE,w,gpus,lr_grid=(.002,.003,.005,.007,.01),refine=True)
    reasons[BIAS_SCALE] = 'eps neighborhood and K=5/10/20; one bounded endpoint refinement; LR at best (eps,C)'

    selected = {}
    for m in SEARCH_METHODS:
        w = best(at(m))
        selected[m] = dict(trial(m,w['lr'],w['C'],w['eps_scale']), source='search_selected', num_bands=4,
                           final_test_top1=w['final_test_top1'], stop_reason=reasons[m],
                           provenance=dict(result_dir=w['result_dir'], result_source=w['source'],
                                           historical_result_dir=w.get('historical_result_dir'),
                                           objective='seed=20261001 epoch-5 final_test_top1',
                                           workload_sha256=w['workload_sha256'], strategy_sha256=w['strategy_sha256']))
    save_json(selected_path, selected)
    search_report()
    return freeze(selected)

def for_scale(method, center, gpus, lr_grid, refine):
    eps = center['eps_scale']
    nearby = list(neighbors(EPS,eps))
    extra_eps = extension(eps,EPS)
    if extra_eps is not None:
        # One outward epsilon point, with K fixed, then its local neighborhood.
        run_stage(method+'_eps_extension',[trial(method,.005,10/extra_eps,extra_eps)],gpus)
        w = best([r for r in at(method,lr=.005) if math.isclose(r['C']*r['eps_scale'],10)])
        if w['eps_scale'] == extra_eps:
            nearby = [eps,extra_eps]
    run_stage(method+'_K', [trial(method,.005,K/e,e) for e in nearby for K in (5,10,20)], gpus)
    w = best(at(method,lr=.005))
    K = w['C'] * w['eps_scale']
    if refine and (math.isclose(K,5) or math.isclose(K,20)):
        extra_K = 2.5 if math.isclose(K,5) else 40
        run_stage(method+'_K_extension',[trial(method,.005,extra_K/w['eps_scale'],w['eps_scale'])],gpus)
        w = best(at(method,lr=.005))
    lr_search(method,w,lr_grid,gpus)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs=3, default=[1,2,3])
    search(parser.parse_args().gpus)
