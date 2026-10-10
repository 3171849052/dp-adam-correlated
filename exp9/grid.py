"""The immutable Cartesian grids; no historical winners or adaptive refinement."""
from itertools import product
from exp9.config import Trial, METHODS, IID, MF, save_json
from exp9 import RESULTS

GRIDS = {
    'cv': {'iid': ((.0003,.0005,.001), (10,30,100), (None,)),
           'mf': ((.001,.003,.005,.007), (3,10,30,100,200), (None,)),
           'scale': ((.001,.003,.005,.007), (30,100,200), (.05,.1,.3))},
    'nlp': {'iid': ((.0005,.001,.002), (.1,.3,1), (None,)),
            'mf': ((.001,.003,.005), (1,10,20), (None,)),
            'scale': ((.001,.003,.005), (100,200,300), (.005,.01,.03,.1))}}

def grid(task):
    jobs = []
    for method in METHODS:
        group = 'iid' if method == IID else 'scale' if method.endswith('-scale') else 'mf'
        jobs.extend(Trial(task,method,lr,C,eps) for lr,C,eps in product(*GRIDS[task][group]))
    assert len(jobs) == (177 if task == 'cv' else 144)
    assert len({j.id for j in jobs}) == len(jobs)
    return jobs

def all_grid(): return grid('cv') + grid('nlp')

def write_grid():
    from exp9.report import write_csv
    for task in ('cv','nlp'):
        rows = [dict(j.values(), trial_id=j.id) for j in grid(task)]
        save_json(RESULTS/f'{task}_grid.json', rows)
        write_csv(RESULTS/f'{task}_grid.csv', rows)

if __name__ == '__main__': write_grid()
