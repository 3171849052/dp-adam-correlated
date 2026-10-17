import csv
import json
from exp14b import ROOT, BASE, RESULTS
from exp14b.config import Trial, IID, MF, file_hash
historical = ROOT/'exp9/results'
METHODS = (IID, *MF)
BASE_LR = dict(zip(METHODS, (.0005,.003,.005,.003)))
FACTORS = (.03,.1,.3)
EPS = (1e-16,1e-12,1e-8)
def jobs(stage='search'):
    grid = [Trial('cv',m,BASE_LR[m]*factor,30,eps=eps,number=i*9+j*3+k+1,seed=20261111,stage=stage)
            for i,m in enumerate(METHODS) for j,factor in enumerate(FACTORS) for k,eps in enumerate(EPS)]
    return grid if stage!='smoke' else [grid[i*9] for i in range(4)]
def frozen_row(task,method):
    with (historical/'frozen_configs.csv').open() as f:
        return next(r for r in csv.DictReader(f) if r['task']==task and r['method']==method)
def reference_dir(cfg):
    stage='final' if cfg.formal else 'search'
    for p in sorted((historical/stage/'cv').glob('*/summary.json')):
        row=json.loads(p.read_text())
        if row['method']==cfg.method and (stage=='search' or row['seed']==cfg.seed):
            return p.parent
    raise ValueError((stage,cfg.method))
def code_hashes():
    return {str(p.relative_to(ROOT)):file_hash(p) for folder in (BASE,ROOT/'exp14a') for p in sorted(folder.glob('*.py'))}
def write_csv(path,rows):
    rows=list(rows)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
