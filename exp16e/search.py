"""Frozen grids and internal-only three-seed selection."""
from dataclasses import replace
import statistics
from exp16e import RESULTS
from exp16e.config import *
from exp16e.support import read,code_hashes
from exp16e.history import config,key,incumbents
CV={
 'momentum-mf-sgd-b2lt':[(.25,3),(.35,3),(.5,5)],
 'momentum-mf-momentum-b2lt':[(3,1),(4,1)],
 'adam3-mf-sgd-b2lt':[(.0005,20),(.0006,20)],
 'adam3-mf-momentum-b2lt':[(.0009,10),(.001,10)]}
NLP_MOM={MOMENTUM[0]:[(.1,.3),(.3,.1),(.3,.3),(.3,1),(1,.1),(1,.3),(2,.3)],**{m:[(.1,5),(.2,10),(.3,10),(.3,20),(.3,40),(.5,20),(.5,40)] for m in MOMENTUM[1:]}}
NLP_ADAM3={
 ADAM3[0]:[(.0002,.3),(.0003,.3),(.0004,.3),(.0003,.1),(.0003,1)],
 ADAM3[1]:[(.0005,10),(.0008,10),(.0012,10),(.0015,10),(.001,20)],
 ADAM3[2]:[(.0005,10),(.001,10),(.0015,10),(.002,10),(.001,20)],
 ADAM3[3]:[(.0005,20),(.001,20),(.0015,20),(.002,20),(.001,40)]}
STPEX_GRID=[(.10,20,.001),(.20,20,.001),(.15,10,.001),(.15,30,.001),(.15,20,.0005),(.15,20,.002),(.20,30,.001),(.20,20,.002)]
NEW_NLP=(STPEX,*MOMENTUM,*ADAM3)

def additions():
    jobs=[Trial('cv',m,lr,c) for m,grid in CV.items() for lr,c in grid]
    old=incumbents()['nlp']
    for m,c in zip(ADAM[1:],(40,10,20,500)):
        jobs.extend(Trial('nlp',m,lr,c,old[m]['eps_scale']) for lr in (.0163,.025))
    base=Trial(**old[ADAM[-1]]);jobs.append(replace(base,eps_scale=.0008))
    jobs.extend(Trial('nlp',STPEX,*v) for v in STPEX_GRID)
    jobs.extend(Trial('nlp',m,lr,c) for m,grid in {**NLP_MOM,**NLP_ADAM3}.items() for lr,c in grid)
    assert len(jobs)==67 and len({key(c) for c in jobs})==67
    return jobs

def freeze_json(name,value):
    path=RESULTS/name
    if path.exists():assert read(path)==value,f'Frozen file changed: {name}'
    else:save_json(path,value)
    return value

def rank(r):return (-r['accuracy'],r['validation_loss'],config(r).id)
def triplet(c,rows):
    index={key(r):r for r in rows if r['status']=='completed'}
    configs=[replace(c,stage='search',seed=SEARCH_SEED)]+[replace(c,stage='review',seed=s) for s in RECHECK_SEEDS]
    if any(key(v) not in index for v in configs):return None
    runs=[index[key(v)] for v in configs]
    assert all(not r['official_evaluation_used'] for r in runs)
    return dict(config=configs[0].values(),mean_accuracy=statistics.mean(r['accuracy'] for r in runs),mean_loss=statistics.mean(r['validation_loss'] for r in runs),runs=[dict(source=r.get('source',str(config(r).output)),trial_id=r['trial_id'],seed=r['seed'],accuracy=r['accuracy'],validation_loss=r['validation_loss']) for r in runs])

def plan_review(rows):
    assert all(not r['official_evaluation_used'] for r in rows)
    successful=[r for r in rows if r['status']=='completed' and r['stage']=='search']
    chosen={};decisions={};old=incumbents();new_keys={key(c) for c in additions()}
    for task,methods in [('cv',CV),('nlp',ADAM[1:]),('nlp',NEW_NLP)]:
        for m in methods:
            candidates=sorted([r for r in successful if r['task']==task and r['method']==m],key=rank)
            label=f'{task}/{m}'
            if task=='nlp' and m in NEW_NLP:
                assert len(candidates)>=2;chosen[label]=[config(r).values() for r in candidates[:2]]
                decisions[label]=dict(rule='top two search-seed candidates',selected=chosen[label])
            else:
                base=Trial(**old[task][m]);inc=next(r for r in candidates if key(r)==key(base))
                new=[r for r in candidates if key(r) in new_keys]
                challenger=new[0] if new else None
                competitive=challenger is not None and rank(challenger)[:2]<=rank(inc)[:2]
                chosen[label]=[config(challenger).values()] if competitive else []
                decisions[label]=dict(rule='best new challenger search accuracy/loss no worse than incumbent on same seed',competitive=competitive,incumbent=base.values(),incumbent_search_accuracy=inc['accuracy'],challenger=None if challenger is None else config(challenger).values())
    jobs=[replace(Trial(**v),stage='review',seed=s) for vs in chosen.values() for v in vs for s in RECHECK_SEEDS]
    assert len(jobs)<=48
    return chosen,decisions,jobs

def select(chosen,decisions,rows):
    assert all(not r['official_evaluation_used'] for r in rows)
    winners=incumbents();evidence={}
    for label,vs in chosen.items():
        task,m=label.split('/');candidates=[]
        if m not in NEW_NLP or task=='cv':
            # Merge every historical three-seed candidate, rather than its single best seed.
            for r in rows:
                if r['task']==task and r['method']==m and r['stage']=='search' and r['status']=='completed':
                    v=triplet(config(r),rows)
                    if v:candidates.append(v)
        else:
            for v in vs:
                result=triplet(Trial(**v),rows)
                if result:candidates.append(result)
        unique={key(Trial(**v['config'])):v for v in candidates};candidates=list(unique.values())
        assert candidates,f'No complete three-seed candidate: {label}'
        candidates.sort(key=lambda r:(-r['mean_accuracy'],r['mean_loss'],Trial(**r['config']).id))
        winners[task][m]=candidates[0]['config'];evidence[label]=dict(decision=decisions[label],candidates=candidates,selected=candidates[0])
    for task,methods in winners.items():
        for m,v in methods.items():
            if f'{task}/{m}' not in evidence:
                three=triplet(Trial(**v),rows);assert three
                evidence[f'{task}/{m}']=dict(rule='unchanged historical freeze',selected=three)
    assert all(set(ms)==set(METHODS) for ms in winners.values())
    return winners,evidence

def authorize(cfg):
    accepted=read(RESULTS/'acceptance.json');assert accepted['status']=='passed' and accepted['code_sha256']==code_hashes()
    if cfg.stage=='final':
        v=read(RESULTS/'final_freeze.json')['winners'][cfg.task][cfg.method]
        assert all(getattr(cfg,k)==v[k] for k in ('lr','clip','eps_scale')) and cfg.seed in FINAL_SEEDS
    else:
        allowed=read(RESULTS/('search_grid.json' if cfg.stage=='search' else 'review_grid.json'))['trials']
        assert cfg.values() in allowed
