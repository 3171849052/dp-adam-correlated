import csv
import json
from dataclasses import replace
from exp14a import ROOT, BASE, RESULTS
from exp14a.config import Trial, file_hash
historical = ROOT/'exp9/results'

def frozen_row(task, method):
    with (historical/'frozen_configs.csv').open() as f:
        return next(r for r in csv.DictReader(f) if r['task']==task and r['method']==method)

def jobs(stage='final'):
    with (historical/'frozen_configs.csv').open() as f:
        return [Trial(r['task'],r['method'],float(r['lr']),float(r['C']),seed=20261111,
                      stage=stage,smoke_phase='final' if stage=='smoke' else 'search')
                for r in csv.DictReader(f) if not r['method'].endswith('-scale')]

def reference_dir(cfg):
    rows=json.loads((historical/'final_results.json').read_text())
    matches=[r for r in rows if r['task']==cfg.task and r['method']==cfg.method and r['seed']==cfg.seed and r['epsilon']==cfg.epsilon]
    assert len(matches)==1
    return historical/'final'/cfg.task/matches[0]['trial_id']

def code_hashes():
    return {p.name:file_hash(p) for p in sorted(BASE.glob('*.py'))}

def write_csv(path,rows):
    rows=list(rows)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
