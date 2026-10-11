"""All 120 internal runs precede selection; 40 final paired runs follow freeze."""
from exp10b.config import (Trial, LAMBDAS, STAGE1_SEEDS, STAGE2_SEEDS, FINAL_SEEDS, COUNTS)

def search_jobs(stage):
    seeds=dict(stage1=STAGE1_SEEDS,stage2=STAGE2_SEEDS)[stage]
    return [Trial(task,lam,seed,stage) for task in ('cv','nlp') for seed in seeds for lam in LAMBDAS]

def choose_winners(rows):
    """Selection consumes internal data only; ties use smallest lambda."""
    expected={j.id:j for stage in ('stage1','stage2') for j in search_jobs(stage)}
    assert len(rows)==len({r['trial_id'] for r in rows})==120
    assert set(expected)=={r['trial_id'] for r in rows}
    for row in rows:
        cfg=expected[row['trial_id']]
        assert all(row[k]==v for k,v in cfg.values().items())
        assert row['status']=='completed' and not row['official_evaluation_used']
        assert row['completed_epochs']==5 and row['optimizer_steps']==cfg.total_steps
    winners={};evidence={}
    for task in ('cv','nlp'):
        groups=[]
        for lam in LAMBDAS:
            group=sorted([r for r in rows if r['task']==task and r['module']==lam],key=lambda r:r['seed'])
            assert [r['seed'] for r in group]==list(STAGE1_SEEDS+STAGE2_SEEDS)
            groups.append(dict(module=lam,mean_internal_accuracy=sum(r['accuracy'] for r in group)/10,
                mean_internal_loss=sum(r['validation_loss'] for r in group)/10,
                seeds=[r['seed'] for r in group],trial_ids=[r['trial_id'] for r in group]))
        rank=lambda r:(-r['mean_internal_accuracy'],r['module'])
        positive=min([r for r in groups if r['module']>0],key=rank)
        global_best=min(groups,key=rank)
        cfg=Trial(task,positive['module'])
        winners[task]=dict(module=positive['module'],lr=cfg.lr,C=cfg.C,
                           mean_internal_accuracy=positive['mean_internal_accuracy'])
        evidence[task]=dict(all_lambdas=groups,best_positive=positive,global_best=global_best,
            selection='10-seed mean final-epoch internal accuracy; exact ties: smaller lambda',
            official_evaluation_affects_selection=False)
    return winners,evidence

def formal_jobs(winners):
    assert set(winners)=={'cv','nlp'}
    assert all(winners[t]['module'] in LAMBDAS[1:] for t in winners)
    return [Trial(task,lam,seed,'final') for task in ('cv','nlp') for seed in FINAL_SEEDS
            for lam in (0.,winners[task]['module'])]

def validate_formal(cfg,winners):
    winner=winners[cfg.task]
    assert cfg.stage=='final'
    assert cfg.module in (0.,winner['module'])
    assert (cfg.lr,cfg.C)==(winner['lr'],winner['C'])

def plan(gpus):
    assert gpus and len(gpus)==len(set(gpus)) and set(gpus)<={1,2,3}
    stages={stage:search_jobs(stage) for stage in ('stage1','stage2')}
    assert {s:len(jobs) for s,jobs in stages.items()}==dict(stage1=36,stage2=84)
    assert len({j.id for jobs in stages.values() for j in jobs})==120
    for cv in LAMBDAS[1:]:
        for nlp in LAMBDAS[1:]:
            assert len(formal_jobs(dict(cv=dict(module=cv),nlp=dict(module=nlp))))==40
    final_slots=[]
    for task in ('cv','nlp'):
        for seed in FINAL_SEEDS:
            for label in (0.,'frozen_best_positive'):
                cfg=Trial(task,0.,seed,'final');slot=cfg.protocol()
                slot['module']=label;slot['bandinvmf']['module']=label
                slot['trial_id']=cfg.id if label==0 else 'resolved_after_stage3_freeze'
                final_slots.append(slot)
    assert len(final_slots)==40
    return dict(counts=COUNTS,gpus=list(gpus),excluded_gpus=[0],
        lambda_values=list(LAMBDAS),cv_per_gpu=1,nlp_per_gpu=2,
        nlp_two_slots_require_stage0_memory_smoke=True,mixed_tasks_per_gpu=False,
        stage1=[dict(j.protocol(),trial_id=j.id) for j in stages['stage1']],
        stage2=[dict(j.protocol(),trial_id=j.id) for j in stages['stage2']],
        stage3=dict(selection='all 120 complete internal runs; choose best positive and report global best',
                    freeze='configuration, code, matrix and selection evidence SHA256 before official data'),
        stage4=dict(tasks=['cv','nlp'],seeds=list(FINAL_SEEDS),
                    lambdas=['0','task-specific frozen best positive'],trials=40,planned_trials=final_slots),
        no_early_elimination=True,full_training_started=False)
