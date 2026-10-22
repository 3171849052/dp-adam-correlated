"""Predeclared eps screening, local joint search, and three-seed selection."""
from dataclasses import replace
import json
import numpy as np
from exp15d import RESULTS
from exp15d.config import Trial, METHODS, EPS_GRID, SEARCH_SEED, FINAL_SEEDS, inherited_eps, file_hash, json_hash, save_json
from exp15d.support import code_hashes
from exp15d.scheduler import run_batch, CORE

FROZEN = RESULTS/'search_plan.json'
SELECTED = RESULTS/'selected_configs.json'
REVIEW = RESULTS/'review_plan.json'
REVIEW_SEEDS = (20261102, 20261103)


def read(path): return json.loads(path.read_text())
def cfg_of(row): return Trial(**{k:row[k] for k in Trial.__dataclass_fields__})
def rank(rows):
    return sorted((r for r in rows if r['status']=='completed'), key=lambda r:(-r['accuracy'],r['validation_loss'],r['trial_id']))


def immutable_json(path, value):
    if path.exists():
        if read(path)!=value: raise ValueError(f'Frozen decision changed: {path}')
    else: save_json(path,value)


def require_phase1():
    acceptance=read(RESULTS/'phase1_acceptance.json');tests=read(RESULTS/'test_results.json')
    if acceptance['status']!='passed' or tests['status']!='passed': raise ValueError('Acceptance required')
    current=code_hashes()
    if acceptance['code_sha256']!=current or tests['code_sha256']!=current: raise ValueError('Code changed after tests/smoke')
    if tests['log_sha256']!=file_hash(RESULTS/'tests.log'): raise ValueError('Test log changed')


def stage1_trials():
    return [Trial(task,method,.5 if task=='cv' else .20 if method=='stpex2' else .15,
                  clip=3. if task=='cv' else 20.,eps_scale=eps,stage='search',seed=SEARCH_SEED)
            for task in ('cv','nlp') for method in METHODS for eps in EPS_GRID]


def joint_trials(retained):
    return [Trial(task,method,lr,clip=clip,eps_scale=eps,stage='search',seed=SEARCH_SEED)
            for task in ('cv','nlp') for method in METHODS for eps in retained[f'{task}:{method}']
            for lr in ((.5,1.,2.) if task=='cv' else (.15,.20,.30))
            for clip in ((1.,3.) if task=='cv' else (1.,10.,20.))]


def retain_eps(rows):
    kept={}
    for task in ('cv','nlp'):
        for method in METHODS:
            historical=inherited_eps(task)
            good=rank([r for r in rows if r['task']==task and r['method']==method and r['eps_scale']!=historical])
            kept[f'{task}:{method}']=[historical]+[r['eps_scale'] for r in good[:2]]
    return kept


def freeze():
    require_phase1()
    value=dict(stage1=[c.values() for c in stage1_trials()],eps_grid=list(EPS_GRID),
        lr_grid={'cv':[.5,1.,2.],'nlp':[.15,.20,.30]},R_grid={'cv':[1.,3.],'nlp':[1.,10.,20.]},
        code_sha256=code_hashes(),maximum_unique_search=195,maximum_review=48,maximum_formal=120,
        selection='internal validation accuracy DESC, cross entropy ASC, trial ID ASC',
        screening='two best nonhistorical eps plus historical eps; stage2 local joint search, not global exhaustive 3D',
        historical_training_reuse=False,official_evaluation_for_selection=False)
    immutable_json(FROZEN,value)
    return value


def load_frozen():
    data=read(FROZEN)
    if data['code_sha256']!=code_hashes(): raise ValueError('Code changed since freeze')
    return data


def authorize(cfg):
    data=load_frozen()
    if cfg.stage=='search':
        allowed=data['stage1']
        if (RESULTS/'joint_plan.json').exists():
            plan=read(RESULTS/'joint_plan.json')
            if plan['search_plan_sha256']!=file_hash(FROZEN): raise ValueError('Joint plan mismatch')
            allowed=allowed+plan['trials']
    else:
        plan=read(REVIEW if cfg.stage=='review' else SELECTED)
        if plan['search_plan_sha256']!=file_hash(FROZEN): raise ValueError('Selection plan mismatch')
        allowed=plan['trials' if cfg.stage=='review' else 'final_trials']
    if cfg.values() not in allowed: raise ValueError('Trial outside authorized frozen plan')


def review_and_select(rows,joint):
    by_id={r['trial_id']:r for r in rows};shortlists={};reviews={}
    for scheme in ('A','B'):
        for task in ('cv','nlp'):
            for method in METHODS:
                candidates=[by_id[c.id] for c in joint if c.task==task and c.method==method and (scheme=='B' or c.clip==1.)]
                good=rank(candidates)
                if not good: raise ValueError(f'No stable candidate: {scheme}/{task}/{method}')
                key=f'{scheme}:{task}:{method}';shortlists[key]=[r['trial_id'] for r in good[:2]]
                for r in good[:2]:
                    for seed in REVIEW_SEEDS:
                        c=replace(cfg_of(r),stage='review',seed=seed);reviews[c.id]=c.values()
    if len(reviews)>48: raise ValueError('Review budget exceeded')
    plan=dict(search_plan_sha256=file_hash(FROZEN),search_results_sha256=file_hash(RESULTS/'search_results.json'),shortlists=shortlists,trials=list(reviews.values()))
    immutable_json(REVIEW,plan)
    reviewed=run_batch('review',[Trial(**v) for v in reviews.values()],allow_nonfinite=True)
    lookup={(r['task'],r['method'],r['lr'],r['clip'],r['eps_scale'],r['seed']):r for r in rows+reviewed}
    rankings={};winners=[];finals={}
    for key,ids in shortlists.items():
        scheme,task,method=key.split(':');aggregates=[]
        for trial_id in ids:
            c=cfg_of(by_id[trial_id]);group=[lookup[(task,method,c.lr,c.clip,c.eps_scale,seed)] for seed in (SEARCH_SEED,)+REVIEW_SEEDS]
            valid=all(r['status']=='completed' and r['completed_epochs']==5 and not r['official_evaluation_used'] for r in group)
            aggregates.append(dict(trial_id=trial_id,lr=c.lr,clip=c.clip,eps_scale=c.eps_scale,valid=valid,
                accuracy_mean=float(np.mean([r['accuracy'] for r in group])) if valid else None,
                loss_mean=float(np.mean([r['validation_loss'] for r in group])) if valid else None,
                seeds=[dict(seed=r['seed'],status=r['status'],accuracy=r.get('accuracy'),loss=r.get('validation_loss')) for r in group]))
        good=sorted((a for a in aggregates if a['valid']),key=lambda a:(-a['accuracy_mean'],a['loss_mean'],a['trial_id']))
        rankings[key]=aggregates
        if not good: raise ValueError(f'No three-seed stable candidate: {key}')
        w=dict(scheme=scheme,task=task,method=method,**good[0]);winners.append(w)
        for seed in FINAL_SEEDS:
            c=replace(cfg_of(by_id[w['trial_id']]),stage='final',seed=seed);finals[c.id]=c.values()
    save_json(RESULTS/'review_results.json',dict(rows=reviewed,rankings=rankings))
    immutable_json(SELECTED,dict(search_plan_sha256=file_hash(FROZEN),review_results_sha256=file_hash(RESULTS/'review_results.json'),
        winners=winners,final_trials=list(finals.values()),official_evaluation_used_for_selection=False))
    return [Trial(**v) for v in finals.values()]


def search_and_review():
    first=run_batch('eps_screening',stage1_trials(),allow_nonfinite=True)
    save_json(RESULTS/'stage1_results.json',first)
    retained=retain_eps(first);joint=joint_trials(retained)
    immutable_json(RESULTS/'joint_plan.json',dict(search_plan_sha256=file_hash(FROZEN),stage1_results_sha256=file_hash(RESULTS/'stage1_results.json'),
        retained_eps=retained,trials=[c.values() for c in joint]))
    # The screening lr/R points recur in stage2; scheduler verifies and reuses them.
    second=run_batch('joint_search',joint,allow_nonfinite=True)
    rows=list({r['trial_id']:r for r in first+second}.values())
    if len(rows)>195: raise ValueError('Search budget exceeded')
    save_json(RESULTS/'search_results.json',rows)
    return review_and_select(rows,joint)
