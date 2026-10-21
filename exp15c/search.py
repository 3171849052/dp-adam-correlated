"""Predeclared extension, deduplicated historical pool, and three-seed review."""
import json
from dataclasses import replace
import numpy as np
from exp15c import RESULTS
from exp15c.config import Trial, METHODS, SEARCH_SEED, FINAL_SEEDS, file_hash, json_hash, save_json
from exp15c import ROOT
from exp15c.scheduler import run_batch, CORE, valid_metrics
from exp15c.support import code_hashes, result_dir
from exp15c.support import read

FROZEN = RESULTS/'extended_grid.json'
SELECTED = RESULTS/'selected_configs.json'
REVIEW = RESULTS/'review_plan.json'
REVIEW_SEEDS = (20261102,20261103)


def rank(rows):
    return sorted((r for r in rows if valid_metrics(r)),
                  key=lambda r:(-r['accuracy'],r['validation_loss'],r['trial_id']))


def cfg_of(row):
    return Trial(**{k:row[k] for k in Trial.__dataclass_fields__})


def require_phase1():
    r=read(RESULTS/'phase1_acceptance.json')
    current=code_hashes()
    if r['status']!='passed' or any(r['code_sha256'][k]!=current[k] for k in CORE):
        raise ValueError('Passing smoke required for unchanged training core')
    tests=read(RESULTS/'test_results.json')
    if tests['status']!='passed' or tests['code_sha256']!=current:
        raise ValueError('Passing tests required for current orchestration code')
    if file_hash(RESULTS/'tests.log')!=tests['log_sha256']: raise ValueError('Test log changed')


def extension_trials():
    old=read(ROOT/'exp15b/results/extended_grid.json')
    jobs=[]
    for task in ('cv','nlp'):
        source=[v for v in old['candidates'].values() if v['task']==task and v['method']=='momentum']
        assert len(source)==(17 if task=='cv' else 22)
        jobs.extend(Trial(task,'momentum-mf-native',v['lr'],v['clip'],stage='search',seed=SEARCH_SEED) for v in source)
        rates=(.001,.01,.05,.1,.3,1.) if task=='cv' else (.001,.003,.01,.03,.1,.3)
        for method,rates_for_method in [('momentum-no-noise',rates),('adam-no-noise',(.00001,.00003,.0001,.0003,.001,.003))]:
            jobs.extend(Trial(task,method,lr,stage='search',seed=SEARCH_SEED) for lr in rates_for_method)
    assert len(jobs)==63 and len({c.id for c in jobs})==63
    return jobs


def freeze():
    require_phase1()
    if FROZEN.exists():return load_frozen()
    jobs=extension_trials()
    schemes={s:{f'{t}:{m}':[] for t in ('cv','nlp') for m in METHODS} for s in ('A','B')}
    for c in jobs:
        key=f'{c.task}:{c.method}'
        schemes['B'][key].append(c.id)
        if not c.private or c.clip==1.: schemes['A'][key].append(c.id)
    frozen=dict(candidates={c.id:c.values() for c in jobs},schemes=schemes,counts=dict(search=63,private_cv=17,private_nlp=22,nonprivate=24),
        code_sha256=code_hashes(),historical_grid_sha256=file_hash(ROOT/'exp15b/results/extended_grid.json'),
        selection_rule='last internal accuracy DESC, loss ASC, trial ID ASC; top 2; three-seed mean accuracy DESC, loss ASC',
        official_evaluation_for_selection=False)
    save_json(FROZEN,frozen)
    (RESULTS/'extended_grid.sha256').write_text(file_hash(FROZEN)+'\n')
    save_json(RESULTS.parent/'configs/grid.json',frozen)
    return frozen


def load_frozen():
    if file_hash(FROZEN)!=(RESULTS/'extended_grid.sha256').read_text().strip():
        raise ValueError('Frozen extension changed')
    data=read(FROZEN);current=code_hashes()
    if any(current[k]!=data['code_sha256'][k] for k in CORE):
        raise ValueError('Training code changed after freeze')
    if file_hash(ROOT/'exp15b/results/extended_grid.json')!=data['historical_grid_sha256']:
        raise ValueError('Historical grid changed')
    return data


def authorize(cfg):
    data=load_frozen()
    if cfg.stage=='search':
        allowed=list(data['candidates'].values())
    elif cfg.stage=='review':
        plan=read(REVIEW)
        if plan['grid_sha256']!=file_hash(FROZEN):raise ValueError('Review/grid mismatch')
        allowed=plan['trials']
    else:
        selected=read(SELECTED)
        if selected['grid_sha256']!=file_hash(FROZEN):raise ValueError('Selection/grid mismatch')
        allowed=selected['final_trials']
    if cfg.values() not in allowed:raise ValueError('Trial outside authorized frozen plan')


def immutable_json(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError(f'Frozen decision changed: {path}')
    else:save_json(path,value)


def search_and_review():
    data=load_frozen()
    jobs=[Trial(**v) for v in data['candidates'].values()]
    rows=run_batch('search',jobs)
    save_json(RESULTS/'search_results.json',rows)
    by_id={r['trial_id']:r for r in rows};shortlists={};reviews={};single_rankings={}
    for scheme,groups in data['schemes'].items():
        for key,ids in groups.items():
            good=rank([by_id[i] for i in ids])
            if not good:raise ValueError(f'No stable candidate: {scheme} {key}')
            decision=f'{scheme}:{key}'
            single_rankings[decision]=[dict(trial_id=r['trial_id'],rank=i+1,accuracy=r['accuracy'],validation_loss=r['validation_loss']) for i,r in enumerate(good)]
            shortlists[decision]=[r['trial_id'] for r in good[:2]]
            for r in good[:2]:
                for seed in REVIEW_SEEDS:
                    c=replace(cfg_of(r),stage='review',seed=seed)
                    reviews[c.id]=c.values()
    plan=dict(grid_sha256=file_hash(FROZEN),shortlists=shortlists,
              single_seed_rankings=single_rankings,trials=list(reviews.values()),
              search_results_sha256=file_hash(RESULTS/'search_results.json'))
    if len(reviews)>32:raise ValueError('Review budget exceeded')
    immutable_json(REVIEW,plan)
    from exp15c.pipeline import status
    status('review',deduplicated_count=len(reviews))
    reviewed=run_batch('review',[Trial(**v) for v in reviews.values()])
    lookup={(r['task'],r['method'],r['lr'],r['clip'],r['seed']):r for r in rows+reviewed}
    winners=[];rankings={};finals={}
    for decision,ids in shortlists.items():
        scheme,task,method=decision.split(':');aggregates=[]
        for trial_id in ids:
            source=by_id[trial_id];c=cfg_of(source)
            measurements=[lookup[(task,method,c.lr,c.clip,seed)] for seed in (SEARCH_SEED,)+REVIEW_SEEDS]
            valid=all(valid_metrics(r) and r['completed_epochs']==5 and not r['official_evaluation_used'] for r in measurements)
            aggregates.append(dict(trial_id=trial_id,lr=c.lr,clip=c.clip,valid=valid,
                accuracy_mean=float(np.mean([r['accuracy'] for r in measurements])) if valid else None,
                accuracy_std=float(np.std([r['accuracy'] for r in measurements],ddof=1)) if valid else None,
                loss_mean=float(np.mean([r['validation_loss'] for r in measurements])) if valid else None,
                seeds=[dict(seed=r['seed'],status=r['status'],accuracy=r.get('accuracy'),loss=r.get('validation_loss'),error=r.get('error')) for r in measurements]))
        good=sorted((r for r in aggregates if r['valid']),key=lambda r:(-r['accuracy_mean'],r['loss_mean'],r['trial_id']))
        if not good:raise ValueError(f'No three-seed stable candidate: {decision}')
        rankings[decision]=dict(ranked=good,excluded=[r for r in aggregates if not r['valid']])
        winner=dict(scheme=scheme,task=task,method=method,**good[0]);winners.append(winner)
        for seed in FINAL_SEEDS:
            c=replace(cfg_of(by_id[winner['trial_id']]),stage='final',seed=seed)
            finals[c.id]=c.values()
    save_json(RESULTS/'review_results.json',dict(rows=reviewed,rankings=rankings,single_seed_rankings=single_rankings))
    selected=dict(grid_sha256=file_hash(FROZEN),winners=winners,final_trials=list(finals.values()),
                  deduplicated_formal_count=len(finals),maximum_formal_count=80,
                  review_results_sha256=file_hash(RESULTS/'review_results.json'),official_evaluation_used_for_selection=False)
    immutable_json(SELECTED,selected)
    return [Trial(**v) for v in finals.values()]
