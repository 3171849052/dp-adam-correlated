"""Pilot-derived immutable grids, common validation selection and paired jobs."""
import json
from dataclasses import replace
from pathlib import Path
from exp15 import ROOT, RESULTS
from exp15.config import Trial, METHODS, SEARCH_SEED, PILOT_SEED, FINAL_SEEDS, file_hash, json_hash, save_json
from exp15.scheduler import run_batch, CORE, completed
from exp15.support import code_hashes

FROZEN=RESULTS/'frozen_grid.json'
SELECTED=RESULTS/'selected_configs.json'


def rank(rows):
    return sorted((r for r in rows if r['status']=='completed'),
                  key=lambda r:(-r['accuracy'],r['validation_loss'],r['trial_id']))


def cfg_of(row):return Trial(**{k:row[k] for k in Trial.__dataclass_fields__})


def require_phase1():
    r=json.loads((RESULTS/'phase1_acceptance.json').read_text())
    if r['status']!='passed':raise ValueError('Phase one has not passed')
    current=code_hashes()
    if any(current[k]!=r['code_sha256'][k] for k in CORE):raise ValueError('Training code changed after acceptance')


def pilot():
    require_phase1()
    # Exp9 Adam learning rates only anchor orders of magnitude, never transfer winners.
    historical=ROOT/'exp9/results/frozen_configs.json'
    old=json.loads(historical.read_text())
    anchors={t:float(old[t]['dp-adam-momentum-bias-bandinvmf']['lr'])*10 for t in ('cv','nlp')}
    jobs=[Trial(t,m,lr,clip,stage='pilot',seed=PILOT_SEED)
          for t in ('cv','nlp') for m in METHODS
          for lr,clip in [(anchors[t]/10,1.),(anchors[t],1.),(anchors[t]*10,1.),(anchors[t],.1),(anchors[t],10.)]]
    save_json(RESULTS/'pilot_plan.json',dict(historical_sha256=file_hash(historical),
        rationale='Three learning-rate orders around 10x the historical Momentum-Bias Adam rate; clip probes 0.1/1/10. Different optimizer update means historical winners are not Exp15 winners.',
        prefix_steps=20,max_boundary_extensions=2,anchors=anchors,trials=[c.values() for c in jobs]))
    rows=run_batch('pilot_initial',jobs,allow_nonfinite=True)
    # Two predeclared boundary checks, before any full-search result exists.
    for round_index in range(2):
        extensions=[]
        for t in ('cv','nlp'):
            for m in METHODS:
                group=[r for r in rows if r['task']==t and r['method']==m and r['clip']==1.]
                good=rank(group)
                if not good:raise ValueError(f'No stable pilot at clip=1: {t} {m}')
                best=good[0]['lr'];tested=sorted({r['lr'] for r in group})
                rate=best/10 if best==tested[0] else best*10 if best==tested[-1] else None
                if rate is not None and rate not in tested:extensions.append(Trial(t,m,rate,1.,stage='pilot',seed=PILOT_SEED))
        if extensions:rows+=run_batch(f'pilot_extension_{round_index+1}',extensions,allow_nonfinite=True)
    save_json(RESULTS/'pilot_results.json',rows)
    return rows


def make_proposal(rows):
    groups=[]
    for t in ('cv','nlp'):
        for m in METHODS:
            group=[r for r in rows if r['task']==t and r['method']==m]
            best=rank([r for r in group if r['clip']==1.])[0]
            lr=best['lr'];rates=[float(f'{lr/factor:.10g}') for factor in (3.,1.,1/3)]
            unstable_clips={r['clip'] for r in group if r['status']=='non_finite'}
            clips=[.1,1.,10.]
            if 10. in unstable_clips:clips=[.1,.3,1.]
            if .1 in unstable_clips:clips=[1.,3.,10.]
            evidence=[]
            for r in group:
                path=cfg_of(r).output/'summary.json'
                diagnostics=cfg_of(r).output/'diagnostics.jsonl'
                metrics=[json.loads(line) for line in diagnostics.read_text().splitlines()] if diagnostics.exists() else []
                evidence.append(dict(lr=r['lr'],clip=r['clip'],status=r['status'],accuracy=r.get('accuracy'),
                    validation_loss=r.get('validation_loss'),summary=str(path.relative_to(ROOT)),sha256=file_hash(path),
                    update_norm=[x['update_norm'] for x in metrics],clip_fraction=[x['clip_fraction'] for x in metrics],
                    denominator_mean=[x['denominator_mean'] for x in metrics]))
            groups.append(dict(task=t,method=m,learning_rates=rates,clips=clips,
                rationale=f'Best stable clip=1 pilot lr={lr:g}; full grid uses lr/3, lr, 3lr. Clip grid includes 1 and coarse observed scales. Pilot uses twenty steps after four-step probes remained near random CV accuracy; this bounds a local search and does not establish an optimum.',
                best_pilot_trial=best['trial_id'],evidence=evidence))
    return dict(groups=groups,selection_rule='last epoch validation accuracy DESC, validation loss ASC, trial ID ASC',
        full_evaluation=dict(epochs=5,cv_search_examples=45000,nlp_search_examples=62000,
                             cv_validation_examples=5000,nlp_validation_examples=5349),
        candidate_budget=dict(A_per_group=3,B_per_group=9,unique_full_search=90),
        official_test_for_selection=False,final_seeds=list(FINAL_SEEDS),
        raw_diagnostics_scope='private local audit artifacts, not DP releases')


def freeze(proposal):
    require_phase1()
    if FROZEN.exists():raise FileExistsError('Grid already frozen; no post-search edits')
    if len(proposal['groups'])!=10:raise ValueError('Missing task-method groups')
    candidates={};schemes={'A':{},'B':{}}
    seen=set()
    for group in proposal['groups']:
        task,method=group['task'],group['method'];key=f'{task}:{method}'
        if (task,method) in seen or task not in ('cv','nlp') or method not in METHODS:raise ValueError('Invalid group')
        seen.add((task,method));rates=group['learning_rates'];clips=group['clips']
        if len(set(rates))!=3 or len(set(clips))!=3 or 1. not in clips:raise ValueError('Unequal grid budget or missing clip=1')
        if not group['rationale'] or not group['evidence']:raise ValueError('Missing pilot rationale')
        for row in group['evidence']:
            p=(ROOT/row['summary']).resolve()
            if not p.is_relative_to(RESULTS/'pilot') or file_hash(p)!=row['sha256']:raise ValueError('Invalid pilot evidence')
        schemes['A'][key]=[];schemes['B'][key]=[]
        for clip in clips:
            for lr in rates:
                cfg=Trial(task,method,lr,clip,stage='search',seed=SEARCH_SEED)
                candidates[cfg.id]=cfg.values();schemes['B'][key].append(cfg.id)
                if clip==1.:schemes['A'][key].append(cfg.id)
    frozen=dict(proposal=proposal,candidates=candidates,schemes=schemes,code_sha256=code_hashes(),
                phase1_sha256=file_hash(RESULTS/'phase1_acceptance.json'))
    frozen['content_sha256']=json_hash(frozen)
    save_json(FROZEN,frozen)
    (RESULTS/'frozen_grid.sha256').write_text(file_hash(FROZEN)+'\n')
    return frozen


def load_frozen():
    if file_hash(FROZEN)!=(RESULTS/'frozen_grid.sha256').read_text().strip():raise ValueError('Frozen grid changed')
    data=json.loads(FROZEN.read_text());current=code_hashes()
    if any(current[k]!=data['code_sha256'][k] for k in CORE):raise ValueError('Training code changed after grid freeze')
    return data


def authorize(cfg):
    data=load_frozen()
    if cfg.stage=='search':
        if cfg.id not in data['candidates'] or cfg.values()!=data['candidates'][cfg.id]:raise ValueError('Trial outside frozen grid')
    else:
        selected=json.loads(SELECTED.read_text())
        if selected['grid_sha256']!=file_hash(FROZEN):raise ValueError('Selection/grid mismatch')
        allowed=selected['final_trials']
        if cfg.values() not in allowed:raise ValueError('Final trial is not a selected configuration')


def search_and_select():
    data=load_frozen()
    rows=run_batch('search',[Trial(**c) for c in data['candidates'].values()],allow_nonfinite=True)
    by_id={r['trial_id']:r for r in rows};winners=[];finals={}
    for scheme,groups in data['schemes'].items():
        for key,ids in groups.items():
            good=rank([by_id[i] for i in ids])
            if not good:raise ValueError(f'No valid candidate: {scheme} {key}')
            winner=good[0];c=cfg_of(winner)
            if winner['completed_epochs']!=5 or winner['official_evaluation_used']:raise ValueError('Invalid selection evaluation')
            winners.append(dict(scheme=scheme,task=c.task,method=c.method,lr=c.lr,clip=c.clip,
                                trial_id=c.id,validation_accuracy=winner['accuracy'],validation_loss=winner['validation_loss']))
            for seed in FINAL_SEEDS:
                final=replace(c,stage='final',seed=seed);finals[final.id]=final.values()
    selected=dict(grid_sha256=file_hash(FROZEN),winners=winners,final_trials=list(finals.values()),
                  deduplicated_formal_count=len(finals),maximum_formal_count=200,
                  result_sources={i:file_hash(cfg_of(r).output/'summary.json') for i,r in by_id.items()})
    if SELECTED.exists() and json.loads(SELECTED.read_text())!=selected:raise ValueError('Frozen selection changed')
    save_json(SELECTED,selected);save_json(RESULTS/'search_results.json',rows)
    return [Trial(**c) for c in finals.values()]
