"""24 initial + six validation-driven local refinements per task/method."""
from dataclasses import replace
import json
from exp10 import RESULTS
from exp10.config import Trial, METHODS, LAMBDAS, BLOCKS, RECHECK_SEEDS, FINAL_SEEDS, save_json
from exp10.launcher import run_queue
PAIRS = {'cv': ((.001,10),(.001,30),(.003,10),(.003,30),(.005,30),(.003,100)),
         'nlp': ((.001,10),(.003,10),(.003,20),(.005,10),(.005,20),(.005,1))}
def initial_jobs():
    return [Trial(task,method,lr,C,module) for task in ('cv','nlp') for method in METHODS
            for module in (LAMBDAS if method==METHODS[0] else BLOCKS) for lr,C in PAIRS[task]]
def rank(rows):
    assert all(r['status']=='completed' for r in rows)
    return sorted(rows,key=lambda r:(-r['accuracy'],r['validation_loss'],r['trial_id']))
def trial(row): return Trial(**{k:row[k] for k in Trial.__dataclass_fields__})
def immutable_json(path,value):
    if path.exists(): assert json.loads(path.read_text())==value, f'Immutable decision changed: {path}'
    else: save_json(path,value)
def refine(rows):
    assert len(rows)==24
    ranked=rank(rows); seen={(r['lr'],r['C'],r['module']) for r in rows}; jobs=[]; decisions=[]
    for anchor in ranked[:2]:
        cfg=trial(anchor); accepted=[]
        proposals=[(cfg.lr*.8,cfg.C),(cfg.lr*1.2,cfg.C),(cfg.lr,cfg.C*.8),
                   (cfg.lr,cfg.C*1.25),(cfg.lr*.9,cfg.C*1.1),(cfg.lr*1.1,cfg.C*.9)]
        for lr,C in proposals:
            lr,C=round(lr,10),round(C,10); key=(lr,C,cfg.module)
            if key in seen: continue
            candidate=replace(cfg,lr=lr,C=C);jobs.append(candidate);seen.add(key);accepted.append(candidate.values())
            if len(accepted)==3: break
        assert len(accepted)==3
        decisions.append(dict(anchor_trial_id=cfg.id,final_internal_accuracy=anchor['accuracy'],
            final_internal_loss=anchor['validation_loss'],candidates=accepted))
    assert len(jobs)==6 and len({j.id for j in jobs})==6
    return jobs,dict(rule='Top two initial candidates; LR x0.8/x1.2, C x0.8; deterministic nearby unused alternatives on collision',
                    selection_data='last-epoch internal validation only',anchors=decisions)
def top_two(rows):
    return {task:{method:[trial(r).values() for r in rank([r for r in rows if r['task']==task and r['method']==method])[:2]]
            for method in METHODS} for task in ('cv','nlp')}
def recheck_jobs(top2):
    return [replace(Trial(**cfg),stage='recheck',seed=seed) for task in ('cv','nlp') for method in METHODS
            for cfg in top2[task][method] for seed in RECHECK_SEEDS]
def choose_winners(top2,rows,rechecks):
    assert len(rows)==120 and len(rechecks)==24
    assert all(r['status']=='completed' for r in rows+rechecks)
    indexed={r['trial_id']:r for r in rows+rechecks}; winners={}; evidence={}
    for task in ('cv','nlp'):
        winners[task]={};evidence[task]={}
        for method in METHODS:
            candidates=[]
            for cfg_values in top2[task][method]:
                cfg=Trial(**cfg_values)
                runs=[indexed[cfg.id]]+[indexed[replace(cfg,stage='recheck',seed=seed).id] for seed in RECHECK_SEEDS]
                candidates.append(dict(lr=cfg.lr,C=cfg.C,module=cfg.module,
                    mean_validation_accuracy=sum(r['accuracy'] for r in runs)/4,
                    mean_validation_loss=sum(r['validation_loss'] for r in runs)/4,
                    base_id=cfg.id,trial_ids=[r['trial_id'] for r in runs]))
            candidates.sort(key=lambda r:(-r['mean_validation_accuracy'],r['mean_validation_loss'],r['base_id']))
            winners[task][method]=candidates[0];evidence[task][method]=candidates
    return winners,evidence
def formal_jobs(winners):
    return [Trial(task,method,**{k:winners[task][method][k] for k in ('lr','C','module')},stage='final',seed=seed)
            for task in ('cv','nlp') for method in METHODS for seed in FINAL_SEEDS]
def search(**schedule):
    initial=initial_jobs(); assert len(initial)==96
    immutable_json(RESULTS/'initial_search_configs.json',[dict(j.values(),trial_id=j.id) for j in initial])
    rows=run_queue(initial,**schedule); save_json(RESULTS/'initial_search_results.json',rows)
    from exp10.audit import audit_pairing
    audit_pairing(rows)
    refinement=[]; decisions={}
    for task in ('cv','nlp'):
        decisions[task]={}
        for method in METHODS:
            jobs,decision=refine([r for r in rows if r['task']==task and r['method']==method])
            refinement+=jobs;decisions[task][method]=decision
    immutable_json(RESULTS/'refinement_decisions.json',decisions)
    immutable_json(RESULTS/'refinement_configs.json',[dict(j.values(),trial_id=j.id) for j in refinement])
    rows+=run_queue(refinement,**schedule)
    assert len(rows)==120 and len({r['trial_id'] for r in rows})==120
    for task in ('cv','nlp'):
        for method in METHODS: assert sum(r['task']==task and r['method']==method for r in rows)==30
    save_json(RESULTS/'search_results.json',rows)
    from exp10.report import write_csv, trial_table
    write_csv(RESULTS/'search_trajectory.csv',trial_table(rows));audit_pairing(rows)
    top2=top_two(rows);immutable_json(RESULTS/'top2.json',top2)
    rechecks=run_queue(recheck_jobs(top2),**schedule);audit_pairing(rows+rechecks)
    save_json(RESULTS/'recheck_results.json',rechecks);write_csv(RESULTS/'recheck_results.csv',trial_table(rechecks))
    winners,evidence=choose_winners(top2,rows,rechecks);immutable_json(RESULTS/'selection_evidence.json',evidence)
    return winners
