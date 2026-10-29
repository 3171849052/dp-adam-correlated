"""Requested coarse grid, top-two review and immutable configuration selection."""
from dataclasses import replace
import numpy as np
from exp16d import RESULTS
from exp16d.config import *
from exp16d.support import read
GRID=[[(.0001,10),(.00015,30),(.0002,30),(.00015,60),(.0003,30)],[(.0001,20),(.0002,20),(.0003,20),(.0002,40),(.0004,20)],[(.0003,10),(.0005,10),(.0007,10),(.0005,30),(.0004,20)],[(.0002,20),(.0004,20),(.0006,20),(.0004,40),(.0004,10)]]
def additions():return [Trial('cv',m,lr,c) for m,grid in zip(METHODS,GRID) for lr,c in grid]
def config(r):return Trial(**{k:r[k] for k in Trial.__dataclass_fields__})
def freeze_json(name,value):
    p=RESULTS/name
    if p.exists():
        if read(p)!=value:raise ValueError(f'Freeze changed: {p}')
    else:save_json(p,value)
    return value
def top_two(rows):
    if any(r['official_evaluation_used'] for r in rows):raise ValueError('Official selection leak')
    return {m:[config(r).values() for r in sorted([r for r in rows if r['method']==m],key=lambda r:(-r['accuracy'],r['validation_loss'],r['trial_id']))[:2]] for m in METHODS}
def select(top,rows):
    index={r['trial_id']:r for r in rows};winners={};evidence={}
    for m,vs in top.items():
        candidates=[]
        for v in vs:
            c=Trial(**v);runs=[index[c.id]]+[index[replace(c,stage='review',seed=s).id] for s in RECHECK_SEEDS]
            assert all(r['status']=='completed' and not r['official_evaluation_used'] for r in runs)
            candidates.append(dict(config=v,mean_accuracy=float(np.mean([r['accuracy'] for r in runs])),mean_loss=float(np.mean([r['validation_loss'] for r in runs])),trial_ids=[r['trial_id'] for r in runs]))
        candidates.sort(key=lambda r:(-r['mean_accuracy'],r['mean_loss'],Trial(**r['config']).id))
        winners[m]=candidates[0]['config'];evidence[m]=dict(candidates=candidates,selected=candidates[0])
    return {'cv':winners},evidence
def controlled():return [Trial('cv',m,.0004,20,seed=s,stage='search' if s==SEARCH_SEED else 'review') for m in METHODS[1:] for s in (SEARCH_SEED,*RECHECK_SEEDS)]
def authorize(cfg):
    from exp16d.stage1 import verify_acceptance
    verify_acceptance(gpu=True)
    if cfg.stage=='final':
        v=read(RESULTS/'final_freeze.json')['winners']['cv'][cfg.method]
        assert all(getattr(cfg,k)==v[k] for k in ('lr','clip','eps_scale')) and cfg.seed in FINAL_SEEDS
    else:
        allowed=read(RESULTS/('search_grid.json' if cfg.stage=='search' else 'review_grid.json'))['trials']
        if cfg.values() not in allowed:raise ValueError('Trial not in frozen grid')
