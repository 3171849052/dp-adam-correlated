"""Predeclared extension, deduplicated historical pool, and three-seed review."""
import json
from dataclasses import replace
import numpy as np
from exp15b import RESULTS
from exp15b.config import Trial, METHODS, SEARCH_SEED, FINAL_SEEDS, file_hash, json_hash, save_json
from exp15b.scheduler import run_batch, CORE
from exp15b.support import code_hashes, result_dir
from exp15b.audit import historical_search, reuse, read

FROZEN = RESULTS/'extended_grid.json'
SELECTED = RESULTS/'selected_configs.json'
REVIEW = RESULTS/'review_plan.json'
REVIEW_SEEDS = (20261102,20261103)


def rank(rows):
    return sorted((r for r in rows if r['status']=='completed'),
                  key=lambda r:(-r['accuracy'],r['validation_loss'],r['trial_id']))


def cfg_of(row):
    return Trial(**{k:row[k] for k in Trial.__dataclass_fields__})


def require_phase1():
    r=read(RESULTS/'phase1_acceptance.json')
    if r['status']!='passed':raise ValueError('Phase one has not passed')
    current=code_hashes()
    if any(current[k]!=r['code_sha256'][k] for k in CORE):
        raise ValueError('Training code changed after acceptance')
    tests=read(RESULTS/'test_results.json')
    if tests['status']!='passed' or tests['log_sha256']!=file_hash(RESULTS/'tests.log'):
        raise ValueError('Passing tests required')


def extension_trials():
    jobs=[]
    for task in ('cv','nlp'):
        for method in METHODS:
            if task=='nlp':
                grid={c:(.15,.20,.30) for c in (1.,5.,10.,20.,40.)}
            elif method in ('momentum','momentum+iid'):
                grid={.3:(2.,3.,4.),1.:(1.,2.,3.,4.),3.:(.5,1.,2.)}
            else:
                grid={.3:(1.,2.,3.),1.:(.4,.6,.8,1.),3.:(.25,.5,.75)}
            jobs += [Trial(task,method,lr,clip,stage='search',seed=SEARCH_SEED)
                     for clip,rates in grid.items() for lr in rates]
    return jobs


def freeze():
    require_phase1()
    if FROZEN.exists():return load_frozen()
    old=historical_search()
    candidates={cfg_of(r).id:cfg_of(r).values() for r in old}
    extension=extension_trials()
    overlap={c.id for c in extension}&set(candidates)
    new=[c for c in extension if c.id not in candidates]
    for c in extension:candidates[c.id]=c.values()
    schemes={s:{f'{t}:{m}':[] for t in ('cv','nlp') for m in METHODS} for s in ('A','B')}
    for trial_id,values in candidates.items():
        c=Trial(**values);key=f'{c.task}:{c.method}'
        schemes['B'][key].append(trial_id)
        if c.clip==1.:schemes['A'][key].append(trial_id)
    counts=dict(extension_cv=sum(c.task=='cv' for c in extension),
                extension_nlp=sum(c.task=='nlp' for c in extension),
                historical_pool=len(old),reusable_overlap=len(overlap),new_search=len(new),
                new_cv=sum(c.task=='cv' for c in new),new_nlp=sum(c.task=='nlp' for c in new),
                combined_pool=len(candidates))
    if counts != dict(extension_cv=60,extension_nlp=90,historical_pool=90,
                      reusable_overlap=16,new_search=134,new_cv=54,new_nlp=80,combined_pool=224):
        raise ValueError(f'Unexpected grid counts: {counts}')
    frozen=dict(candidates=candidates,schemes=schemes,counts=counts,
                extension_ids=[c.id for c in extension],historical_ids=[cfg_of(r).id for r in old],
                code_sha256=code_hashes(),phase1_sha256=file_hash(RESULTS/'phase1_acceptance.json'),
                source_audit_sha256=file_hash(RESULTS/'source_audit.json'),
                selection_rule='last internal validation accuracy DESC, loss ASC, trial ID ASC',
                review_rule='top two per task/method/scheme; deduplicate; add seeds 20261102/20261103; choose three-seed mean accuracy DESC, mean loss ASC, trial ID ASC',
                official_evaluation_for_selection=False,maximum_review_trainings=96,maximum_main_formal_trainings=240,
                controlled_comparison='all matching lr/clip search pairs; three-seed pairs where both reviewed; ten formal seeds only if identical selected configs. Do not label n=1 validation effects as ten-seed official evidence.',
                privacy_scope='epsilon=8, delta=1e-5 per training only; no composed search guarantee')
    frozen['content_sha256']=json_hash(frozen)
    save_json(FROZEN,frozen)
    (RESULTS/'extended_grid.sha256').write_text(file_hash(FROZEN)+'\n')
    save_json(RESULTS.parent/'configs/extended_grid.json',dict(extension=[c.values() for c in extension],counts=counts))
    return frozen


def load_frozen():
    if file_hash(FROZEN)!=(RESULTS/'extended_grid.sha256').read_text().strip():
        raise ValueError('Frozen extension changed')
    data=read(FROZEN);current=code_hashes()
    if any(current[k]!=data['code_sha256'][k] for k in CORE):
        raise ValueError('Training code changed after freeze')
    if file_hash(RESULTS/'source_audit.json')!=data['source_audit_sha256']:
        raise ValueError('Historical audit changed after freeze')
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
    for c in jobs:reuse(c)
    rows=run_batch('search',jobs,allow_nonfinite=True)
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
    if len(reviews)>96:raise ValueError('Review budget exceeded')
    immutable_json(REVIEW,plan)
    reviewed=run_batch('review',[Trial(**v) for v in reviews.values()],allow_nonfinite=True)
    lookup={(r['task'],r['method'],r['lr'],r['clip'],r['seed']):r for r in rows+reviewed}
    winners=[];rankings={};finals={}
    for decision,ids in shortlists.items():
        scheme,task,method=decision.split(':');aggregates=[]
        for trial_id in ids:
            source=by_id[trial_id];c=cfg_of(source)
            measurements=[lookup[(task,method,c.lr,c.clip,seed)] for seed in (SEARCH_SEED,)+REVIEW_SEEDS]
            valid=all(r['status']=='completed' and r['completed_epochs']==5 and not r['official_evaluation_used'] for r in measurements)
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
                  deduplicated_formal_count=len(finals),maximum_formal_count=240,
                  review_results_sha256=file_hash(RESULTS/'review_results.json'),official_evaluation_used_for_selection=False)
    immutable_json(SELECTED,selected)
    return [Trial(**v) for v in finals.values()]
