"""Full grids, final-epoch Top-2, and independent three-seed winner selection."""
from dataclasses import replace
from exp9 import RESULTS
from exp9.config import Trial, METHODS, RECHECK_SEEDS, save_json
from exp9.grid import all_grid
from exp9.launcher import run_queue

def rank(rows):
    return sorted((r for r in rows if r['status']=='completed'), key=lambda r:(-r['accuracy'],r['trial_id']))

def top_two(rows):
    selected={}
    for task in ('cv','nlp'):
        selected[task]={}
        for method in METHODS:
            candidates=rank([r for r in rows if r['task']==task and r['method']==method])
            assert len(candidates)>=2, f'Not enough successful candidates: {task}/{method}'
            selected[task][method]=[{k:row[k] for k in Trial.__dataclass_fields__} for row in candidates[:2]]
    return selected

def recheck_jobs(top2):
    return [replace(Trial(**cfg),stage='recheck',seed=seed) for task in ('cv','nlp')
            for method in METHODS for cfg in top2[task][method] for seed in RECHECK_SEEDS]

def choose_winners(top2,search_rows,recheck_rows):
    assert len(recheck_rows)==56 and all(r['status']=='completed' for r in recheck_rows)
    indexed={r['trial_id']:r for r in search_rows+recheck_rows};winners={};evidence={}
    for task in ('cv','nlp'):
        winners[task]={};evidence[task]={}
        for method in METHODS:
            candidates=[]
            for values in top2[task][method]:
                cfg=Trial(**values)
                runs=[indexed[cfg.id]]+[indexed[replace(cfg,stage='recheck',seed=seed).id] for seed in RECHECK_SEEDS]
                assert all(r['status']=='completed' for r in runs)
                candidates.append(dict(lr=cfg.lr,C=cfg.C,eps_scale=cfg.eps_scale,
                                       mean_validation_accuracy=sum(r['accuracy'] for r in runs)/3,
                                       trial_ids=[r['trial_id'] for r in runs],base_id=cfg.id))
            candidates.sort(key=lambda r:(-r['mean_validation_accuracy'],r['base_id']))
            winners[task][method]=candidates[0];evidence[task][method]=candidates
    return winners,evidence

def search(**schedule):
    from exp9.report import write_csv, trial_table
    rows=run_queue(all_grid(),**schedule)
    assert len(rows)==321
    write_csv(RESULTS/'grid_results.csv',trial_table(rows));save_json(RESULTS/'grid_results.json',rows)
    for task in ('cv','nlp'): write_csv(RESULTS/f'{task}_grid_results.csv',trial_table([r for r in rows if r['task']==task]))
    top2=top_two(rows);save_json(RESULTS/'top2.json',top2)
    rechecks=run_queue(recheck_jobs(top2),**schedule)
    write_csv(RESULTS/'recheck_results.csv',trial_table(rechecks));save_json(RESULTS/'recheck_results.json',rechecks)
    winners,evidence=choose_winners(top2,rows,rechecks)
    save_json(RESULTS/'selection_evidence.json',evidence)
    return winners

if __name__=='__main__':
    import argparse
    from exp9.audit import verify_stage1
    from exp9.frozen import freeze
    from exp9.grid import write_grid
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    args=p.parse_args();verify_stage1();write_grid();freeze(search(gpus=args.gpus))
