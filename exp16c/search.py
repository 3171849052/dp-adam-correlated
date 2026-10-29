"""Only requested additions; three-seed selection with incumbent protection."""
from dataclasses import replace
import numpy as np
from exp16c import RESULTS
from exp16c.config import *
from exp16c.support import read

def additions():
    jobs=[Trial('cv',m,.003,c) for m in A_METHODS for c in (12.,24.)]
    jobs += [Trial('cv',STPEX,lr,c,e) for lr,c,e in [(.10,10.,3.),(.05,10.,3.),(.15,20.,3.),(.15,10.,6.),(.10,20.,3.),(.10,20.,6.)]]
    jobs += [Trial('cv',m,lr,c) for m in MOMENTUM_METHODS for lr,c in [(.5,3.),(1.,1.),(1.,3.),(2.,.3),(2.,1.)]]
    return jobs

def config(row):return Trial(**{k:row[k] for k in Trial.__dataclass_fields__})
def rank(rows):
    if any(r.get('official_evaluation_used') for r in rows):raise ValueError('Official selection leak')
    return sorted([r for r in rows if r['status']=='completed'],key=lambda r:(-r['accuracy'],r['validation_loss'],r['trial_id']))
def top_two(rows):
    out={}
    for m in TUNED_METHODS:
        candidates=rank([r for r in rows if r['method']==m])
        if len(candidates)<2:raise ValueError(f'Insufficient valid candidates: {m}')
        out[m]=[config(r).values() for r in candidates[:2]]
    return out

def freeze_json(name,value):
    p=RESULTS/name
    if p.exists():
        if read(p)!=value:raise ValueError(f'Freeze changed: {p}')
    else:save_json(p,value)
    return value

def select(top,rows,incumbents):
    index={r['trial_id']:r for r in rows};winners=dict(incumbents);evidence={}
    for m,configs in top.items():
        # An old three-seed winner remains the baseline even if displaced from single-seed top two.
        pool=list(configs)
        if m in A_METHODS and incumbents[m] not in pool:pool.append(incumbents[m])
        candidates=[]
        for v in pool:
            c=Trial(**v);ids=[c.id]+[replace(c,stage='review',seed=s).id for s in RECHECK_SEEDS]
            runs=[index[i] for i in ids]
            if not all(r['status']=='completed' for r in runs):continue
            if any(r['official_evaluation_used'] for r in runs):raise ValueError('Official selection leak')
            candidates.append(dict(config=v,mean_accuracy=float(np.mean([r['accuracy'] for r in runs])),mean_loss=float(np.mean([r['validation_loss'] for r in runs])),trial_ids=ids))
        candidates.sort(key=lambda r:(-r['mean_accuracy'],r['mean_loss'],Trial(**r['config']).id))
        if not candidates:raise ValueError(f'No valid three-seed candidate: {m}')
        chosen=candidates[0]
        if m in A_METHODS:
            old=next(r for r in candidates if r['config']==incumbents[m])
            if (-chosen['mean_accuracy'],chosen['mean_loss'])>=(-old['mean_accuracy'],old['mean_loss']):chosen=old
        winners[m]=chosen['config'];evidence[m]=dict(candidates=candidates,selected=chosen,incumbent_retained=m in A_METHODS and chosen['config']==incumbents[m])
    return {'cv':winners},evidence

def authorize(cfg):
    from exp16c.stage1 import verify_acceptance
    verify_acceptance()
    if cfg.stage=='final':
        v=read(RESULTS/'final_freeze.json')['winners']['cv'][cfg.method]
        if any(getattr(cfg,k)!=v[k] for k in ('lr','clip','eps_scale')) or cfg.seed not in FINAL_SEEDS:raise ValueError('Unfrozen formal trial')
    else:
        allowed=read(RESULTS/('search_grid.json' if cfg.stage=='search' else 'review_grid.json'))['trials']
        if cfg.values() not in allowed:raise ValueError('Trial not in frozen grid')
