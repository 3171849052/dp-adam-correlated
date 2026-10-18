import csv
import json
from functools import lru_cache
from exp14c import ROOT, BASE, RESULTS
from exp14c.config import Trial, METHODS, IID, FINAL_SEEDS, file_hash
historical = ROOT/'exp9/results'

def grid(task):
    jobs=[]
    for method in METHODS:
        if task=='cv':
            rates=(.00010,.00015,.00020,.00025,.00030) if method==IID else (.00020,.00030,.00040,.00050,.00060)
            clips=(10,30,100)
        else:
            rates=(.00010,.00020,.00030,.00050,.00080) if method==IID else (.00015,.00030,.00050,.00080,.00100)
            clips=(.1,.3,1) if method==IID else (10,20,40)
        jobs.extend(Trial(task,method,lr,C) for lr in rates for C in clips)
    assert len(jobs)==60 and len({c.id for c in jobs})==60
    return jobs

def smoke_jobs():
    from dataclasses import replace
    return [replace(next(c for c in grid(task) if c.method==method),stage='smoke',seed=20261111,
                    smoke_phase=phase)
            for task in ('cv','nlp') for method in METHODS
            for phase in (('search','final') if task=='cv' else ('search',))]

@lru_cache(None)
def reference_dir(cfg):
    stage='final' if cfg.formal else 'search'
    for p in sorted((historical/stage/cfg.task).glob('*/summary.json')):
        r=json.loads(p.read_text())
        if r['method']==cfg.method and r['status']=='completed' and (stage=='search' or r['seed']==cfg.seed):
            return p.parent
    raise ValueError((stage,cfg.task,cfg.method,cfg.seed))

def frozen_row(task,method):
    return next(r for r in json.loads((RESULTS/'frozen_manifest.json').read_text())['winners'] if r['task']==task and r['method']==method)

def code_hashes():
    return {str(p.relative_to(ROOT)):file_hash(p) for folder in (BASE,ROOT/'exp14a') for p in sorted(folder.glob('*.py'))}

def write_csv(path,rows):
    rows=list(rows)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)

def protected_snapshot():
    roots=[p for p in ROOT.glob('exp*') if p.is_dir() and p!=BASE]+[ROOT/'data',ROOT/'cache']
    return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
            for root in roots for p in root.rglob('*') if p.is_file()}
