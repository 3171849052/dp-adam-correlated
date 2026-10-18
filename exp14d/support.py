import csv
import json
from functools import lru_cache
from dataclasses import replace
from exp14d import ROOT, BASE, RESULTS
from exp14d.config import Trial, METHODS, IID, MF, ADAM3, FINAL_SEEDS, file_hash
historical = ROOT/'exp9/results'

def grid(task):
    rates = ((.0001,.0003,.0005),(.0005,.001,.003),(.0005,.001,.003),(.0001,.0002,.0004),(.0002,.0004,.0006),(.0002,.0004,.0006)) if task=='cv' else ((.0005,.001,.002),(.001,.003,.005),(.001,.003,.005),(.0003,.0008,.0015),(.0003,.001,.0015),(.0003,.001,.0015))
    return [Trial(task,m,lr,1,eps_scale=(.05 if task=='cv' else .01) if m.endswith('-scale') else None) for m,rs in zip(METHODS,rates) for lr in rs]

def boundary_grid():
    return [Trial(t,ADAM3[i],lr,C,module='A') for t,i,rs,C in [('cv',0,(.0001,.00015),30),('cv',1,(.0007,.0008),30),('nlp',1,(.00125,.0015),10),('nlp',2,(.00125,.0015),20)] for lr in rs]

def smoke_jobs():
    return [replace(next(c for c in grid(t) if c.method==m),stage='smoke',seed=20261111,smoke_phase=phase) for t in ('cv','nlp') for m in METHODS for phase in (('search','final') if t=='cv' else ('search',))]

@lru_cache(None)
def reference_dir(cfg):
    stage='final' if cfg.formal else 'search'
    for p in sorted((historical/stage/cfg.task).glob('*/summary.json')):
        r=json.loads(p.read_text())
        if r['method']==cfg.historical_method and r['status']=='completed' and (stage=='search' or r['seed']==cfg.seed): return p.parent
    raise ValueError(cfg)

def frozen_row(task,method,module):
    return next(r for r in json.loads((RESULTS/f'{module}_frozen_manifest.json').read_text())['winners'] if r['task']==task and r['method']==method)

def code_hashes():
    return {str(p.relative_to(ROOT)):file_hash(p) for folder in (BASE,ROOT/'exp14a',ROOT/'exp14c',ROOT/'exp9') for p in sorted(folder.glob('*.py'))}

def write_csv(path,rows):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)

def protected_snapshot():
    roots=[p for p in ROOT.glob('exp*') if p.is_dir() and p!=BASE]+[ROOT/'data',ROOT/'cache']
    return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns] for root in roots for p in root.rglob('*') if p.is_file()}

def historical_rows(experiment,kind):
    path=ROOT/experiment/'results'/('formal_results.json' if experiment=='exp14c' and kind=='final' else kind+'_results.json')
    rows=[json.loads(p.read_text()) for p in sorted((ROOT/experiment/"results/search").glob("*/*/summary.json"))] if experiment=="exp9" and kind=="search" else json.loads(path.read_text())
    return [dict(r,source_experiment=experiment,source_summary=str(ROOT/experiment/'results'/r['result_dir']/'summary.json'),method=r['method']+'-adam3' if experiment=='exp14c' else r['method']) for r in rows]
