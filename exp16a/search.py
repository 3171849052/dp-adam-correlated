"""Predeclared coarse/fine grids, internal-only rankings, three-seed selection."""
from itertools import product
from dataclasses import replace
import numpy as np
from exp16a import RESULTS
from exp16a.config import Trial,METHODS,SEARCH_SEED,RECHECK_SEEDS,FINAL_SEEDS,save_json,file_hash,json_hash
from exp16a.support import read,code_hashes

def category(method):return 'adam' if method=='adam' else 'iid' if method=='adam-iid' else 'stp' if '-stp-' in method else 'mf'
GRIDS={
 'cv':dict(adam=dict(lr=[1e-4,3e-4,1e-3]),iid=dict(lr=[3e-4,5e-4,1e-3],clip=[10.,30.,100.]),
     mf=dict(lr=[1e-3,3e-3,7e-3],clip=[10.,30.,100.]),stp=dict(lr=[1e-3,3e-3,7e-3],clip=[30.,200.,300.],eps_scale=[.01,.05,.3])),
 'nlp':dict(adam=dict(lr=[3e-4,1e-3,2e-3]),iid=dict(lr=[5e-4,2e-3,3e-3],clip=[.1,.3,1.]),
     mf=dict(lr=[1e-3,3e-3,7e-3],clip=[1.,10.,40.]),stp=dict(lr=[1e-3,3e-3,7e-3],clip=[100.,300.,500.],eps_scale=[.003,.01,.1]))}
def expand(task,method,grid):
    keys=list(grid)
    return [Trial(task,method,**dict(zip(keys,values))) for values in product(*(grid[k] for k in keys))]
def coarse_jobs():
    # Interleave task/method so mixed scheduling is exercised from the beginning.
    groups=[expand(task,m,GRIDS[task][category(m)]) for m in METHODS for task in ('cv','nlp')]
    return [g[i] for i in range(27) for g in groups if i<len(g)]
def rank(rows):
    if any(r.get('official_evaluation_used') for r in rows):raise ValueError('Official data forbidden in selection')
    return sorted((r for r in rows if r['status']=='completed'),key=lambda r:(-r['accuracy'],r['validation_loss'],r['trial_id']))
def refine(values,best):
    """Log-scale half-intervals; edge winners extend one half-interval outward."""
    values=sorted(values);i=values.index(best)
    lo=np.log(values[max(0,i-1)]);mid=np.log(best);hi=np.log(values[min(len(values)-1,i+1)])
    if i==0:lo=mid-(hi-mid)
    if i==len(values)-1:hi=mid+(mid-lo)
    # Retain the winner exactly so duplicate success can be reused.
    return [float(f'{np.exp((lo+mid)/2):.12g}'),best,float(f'{np.exp((mid+hi)/2):.12g}')]
def fine_jobs(coarse_rows):
    jobs=[];basis={}
    for task in ('cv','nlp'):
        basis[task]={}
        for m in METHODS:
            best=rank([r for r in coarse_rows if r['task']==task and r['method']==m])[0]
            grid={k:refine(v,best[k]) for k,v in GRIDS[task][category(m)].items()}
            basis[task][m]=dict(coarse_winner=best['trial_id'],grid=grid)
            jobs.extend(expand(task,m,grid))
    return jobs,basis
def top_two(rows):
    out={}
    for task in ('cv','nlp'):
        out[task]={}
        for m in METHODS:
            candidates=rank([r for r in rows if r['task']==task and r['method']==m])
            if len(candidates)<2:raise RuntimeError(f'Fewer than two valid configurations: {task}/{m}')
            out[task][m]=[{k:r[k] for k in Trial.__dataclass_fields__} for r in candidates[:2]]
    return out
def review_jobs(top):return [replace(Trial(**v),stage='review',seed=s) for task in top.values() for configs in task.values() for v in configs for s in RECHECK_SEEDS]
def select(top,search,review):
    index={r['trial_id']:r for r in search+review};winners={};evidence={}
    for task in top:
        winners[task]={};evidence[task]={}
        for m,configs in top[task].items():
            candidates=[]
            for v in configs:
                c=Trial(**v);runs=[index[c.id]]+[index[replace(c,stage='review',seed=s).id] for s in RECHECK_SEEDS]
                if not all(r['status']=='completed' for r in runs):continue
                if any(r['official_evaluation_used'] for r in runs):raise ValueError('Official selection leak')
                candidates.append(dict(config=v,mean_accuracy=float(np.mean([r['accuracy'] for r in runs])),mean_loss=float(np.mean([r['validation_loss'] for r in runs])),trial_ids=[r['trial_id'] for r in runs],base_id=c.id))
            candidates.sort(key=lambda r:(-r['mean_accuracy'],r['mean_loss'],r['base_id']))
            if not candidates:raise RuntimeError(f'No three-seed valid configuration: {task}/{m}')
            winners[task][m]=candidates[0]['config'];evidence[task][m]=candidates
    return winners,evidence
def freeze_json(name,value):
    p=RESULTS/name
    if p.exists():
        if read(p)!=value:raise ValueError(f'Frozen file changed: {p}')
    else:save_json(p,value)
    return value
def authorize(cfg):
    from exp16a.stage1 import verify_acceptance
    verify_acceptance()
    if cfg.stage=='final':
        frozen=read(RESULTS/'final_freeze.json');v=frozen['winners'][cfg.task][cfg.method]
        if any(getattr(cfg,k)!=v[k] for k in ('lr','clip','eps_scale')):raise ValueError('Final hyperparameters not frozen')
        if cfg.seed not in FINAL_SEEDS:raise ValueError('Wrong formal seed')
    else:
        names=('coarse_grid.json','fine_grid.json') if cfg.stage=='search' else ('review_grid.json',)
        allowed={json_hash(v) for name in names if (RESULTS/name).exists() for v in read(RESULTS/name)['trials']}
        if json_hash(cfg.values()) not in allowed:raise ValueError('Training not in frozen grid')
