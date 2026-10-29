"""Independent eleven-point grids, at most two outward points, three-seed selection."""
from dataclasses import replace
import statistics
from exp16f import RESULTS
from exp16f.config import *
from exp16f.support import read,code_hashes
from exp16f.history import config,key
CENTERS={
 'cv':[(.003,100,.05),(.003,200,.05),(.0004,100,.05),(.0004,200,.05),(.0006,100,.05),(.0004,200,.05)],
 'nlp':[(.0106926766,200,.00164317),(.0106926766,500,.00164317),(.0015,100,.00164317),(.0015,500,.00164317),(.002,200,.00164317),(.0015,500,.00164317)]}
def centers():return [Trial(task,m,*v) for task,vs in CENTERS.items() for m,v in zip(METHODS,vs)]
def additions():
    return [replace(c,lr=c.lr*l,clip=c.clip*k,eps_scale=c.eps_scale*e) for c in centers() for l,k,e in [(0.5,1,1),(.75,1,1),(1,1,1),(1.5,1,1),(2,1,1),(1,.3,1),(1,3,1),(1,1,.2),(1,1,5),(1.5,.3,1),(.75,3,1)]]
def freeze_json(name,value):
    p=RESULTS/name
    if p.exists():assert read(p)==value,f'Frozen file changed: {name}'
    else:save_json(p,value)
    return value
def rank(r):return (-r['accuracy'],r['validation_loss'],config(r).id)
def candidates(rows,c):
    assert all(not r['official_evaluation_used'] for r in rows)
    return sorted([r for r in rows if r['task']==c.task and r['method']==c.method and r['stage']=='search' and r['status']=='completed'],key=rank)
def boundary(rows):
    jobs=[];evidence={}
    for c in centers():
        best=config(candidates(rows,c)[0]);extra=[]
        # Deterministic priority lr, clip, eps; extend each active direction once.
        for field,lo,hi in [('lr',.5,2),('clip',.3,3),('eps_scale',.2,5)]:
            ratio=getattr(best,field)/getattr(c,field)
            if abs(ratio-lo)<1e-10 or abs(ratio-hi)<1e-10:
                extra.append(replace(best,**{field:getattr(best,field)*(.5 if abs(ratio-lo)<1e-10 else 2)}))
        jobs.extend(extra[:2]);evidence[c.task+'/'+c.method]=dict(best=best.values(),extensions=[v.values() for v in extra[:2]])
    return jobs,evidence

def triplet(c,rows):
    idx={key(r):r for r in rows if r['status']=='completed'}
    cs=[replace(c,stage='search',seed=SEARCH_SEED)]+[replace(c,stage='review',seed=s) for s in RECHECK_SEEDS]
    if any(key(v) not in idx for v in cs):return None
    rs=[idx[key(v)] for v in cs];assert all(not r['official_evaluation_used'] for r in rs)
    return dict(config=cs[0].values(),mean_accuracy=statistics.mean(r['accuracy'] for r in rs),mean_loss=statistics.mean(r['validation_loss'] for r in rs),runs=[dict(trial_id=r['trial_id'],seed=r['seed'],accuracy=r['accuracy'],validation_loss=r['validation_loss']) for r in rs])
def plan_review(rows):
    chosen={};jobs=[]
    for c in centers():
        top=candidates(rows,c)[:2];assert len(top)==2
        chosen[c.task+'/'+c.method]=[config(r).values() for r in top]
        jobs.extend(replace(config(r),stage='review',seed=s) for r in top for s in RECHECK_SEEDS)
    return chosen,jobs

def select(chosen,rows):
    winners={'cv':{},'nlp':{}};evidence={}
    for label,vs in chosen.items():
        rs=[triplet(Trial(**v),rows) for v in vs];assert all(rs),'Incomplete three-seed review'
        rs.sort(key=lambda r:(-r['mean_accuracy'],r['mean_loss'],Trial(**r['config']).id))
        task,m=label.split('/');winners[task][m]=rs[0]['config'];evidence[label]=dict(candidates=rs,selected=rs[0])
    return winners,evidence

def authorize(cfg):
    a=read(RESULTS/'acceptance.json');assert a['status']=='passed' and a['code_sha256']==code_hashes()
    if cfg.stage=='final':
        v=read(RESULTS/'final_freeze.json')['winners'][cfg.task][cfg.method]
        assert all(getattr(cfg,k)==v[k] for k in ('lr','clip','eps_scale')) and cfg.seed in FINAL_SEEDS
    else:
        names=('search_grid.json','boundary_grid.json') if cfg.stage=='search' else ('review_grid.json',)
        assert any((RESULTS/n).exists() and cfg.values() in read(RESULTS/n)['trials'] for n in names)
