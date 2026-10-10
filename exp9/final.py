"""Formal runs and fixed-hyperparameter epsilon scan, dependent on the freeze."""
from exp9 import RESULTS
from exp9.config import Trial, METHODS, FINAL_SEEDS, EPSILONS, save_json
from exp9.launcher import run_queue

def formal_jobs(winners,stage='final'):
    seeds=FINAL_SEEDS if stage=='final' else FINAL_SEEDS[:3]
    epsilons=(8,) if stage=='final' else EPSILONS
    return [Trial(task,method,**{k:winners[task][method][k] for k in ('lr','C','eps_scale')},
                  seed=seed,epsilon=epsilon,stage=stage)
            for task in ('cv','nlp') for epsilon in epsilons for seed in seeds for method in METHODS]

def run_formal(stage='final',**schedule):
    import json
    from exp9.frozen import load_frozen
    from exp9.audit import audit_trial
    from exp9.report import write_csv,trial_table
    if stage=='sweep':
        final_rows=json.loads((RESULTS/'final_results.json').read_text())
        assert len(final_rows)==140 and all(r['status']=='completed' for r in final_rows)
        expected=formal_jobs(load_frozen(),'final')
        assert {r['trial_id'] for r in final_rows}=={j.id for j in expected}
        for job in expected: audit_trial(job)
    jobs=formal_jobs(load_frozen(),stage);assert len(jobs)==(140 if stage=='final' else 126)
    rows=run_queue(jobs,**schedule)
    assert all(r['status']=='completed' for r in rows)
    save_json(RESULTS/f'{stage}_results.json',rows);write_csv(RESULTS/f'{stage}_raw.csv',trial_table(rows))
    return rows

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('final','sweep'),default='final')
    p.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3]);a=p.parse_args()
    from exp9.audit import verify_stage1
    from exp9.data import prepare_official
    verify_stage1();prepare_official();run_formal(a.stage,gpus=a.gpus)
