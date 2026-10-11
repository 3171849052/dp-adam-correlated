"""Exactly 120 search, 24 recheck and 40 formal trainings on physical GPUs 1/2/3."""
import argparse
import fcntl
import json
import time
import traceback
from exp10 import BASE,RESULTS
from exp10.config import COUNTS,save_json

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',type=int,nargs='+',default=[1,2,3],choices=(1,2,3))
    p.add_argument('--plan',action='store_true',help='Validate/show budget and initial grid; launch no training')
    return p
def plan(gpus):
    from exp10.search import initial_jobs,PAIRS
    from exp10.config import METHODS,LAMBDAS,BLOCKS,RECHECK_SEEDS,FINAL_SEEDS
    assert len(gpus)==len(set(gpus)) and gpus and set(gpus)<={1,2,3}
    jobs=initial_jobs();assert len(jobs)==len({j.id for j in jobs})==96
    for task in ('cv','nlp'):
        assert len(PAIRS[task])==6 and len({lr for lr,C in PAIRS[task]})>=3 and len({C for lr,C in PAIRS[task]})>=3
        assert ((.003,30) if task=='cv' else (.005,20)) in PAIRS[task]
        for method in METHODS: assert sum(j.task==task and j.method==method for j in jobs)==24
    assert 4*(24+6)+4*2*3+4*10==184
    return dict(counts=COUNTS,gpus=gpus,excluded_gpus=[0],cv_per_gpu=1,nlp_per_gpu=2,mixed_tasks_per_gpu=False,
        epochs_per_trial=5,epsilon=8.,delta=1e-5,logical_batch=1000,cv_search_T=225,cv_final_T=250,nlp_T=310,
        search_seed=20261101,recheck_seeds=list(RECHECK_SEEDS),final_seeds=list(FINAL_SEEDS),
        initial_pairs=PAIRS,module_candidates=dict(zip(METHODS,(LAMBDAS,BLOCKS))),
        initial_trials=[dict(j.values(),trial_id=j.id) for j in jobs],
        deferred_slots=dict(refinement=24,recheck=24,final=40),
        adaptive_rule='Six distinct local refinements per cell from its initial internal-validation Top-2',
        stage_order=['initial_search','refinement','recheck','freeze','final','report'])
def main():
    args=parser().parse_args();protocol=plan(args.gpus)
    if args.plan: print(json.dumps(protocol,indent=2));return
    from exp10.audit import verify_stage1,full_audit,protected_snapshot
    from exp10.search import search,formal_jobs
    from exp10.frozen import freeze
    from exp10.data import prepare_official
    from exp10.launcher import run_queue
    from exp10.report import generate_report,write_csv,trial_table
    lock=(BASE/'runtime/stage2.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_stage1()
    assert protected_snapshot()==json.loads((RESULTS/'protected_files_before.json').read_text())
    save_json(RESULTS/'stage2_plan.json',protocol)
    def status(stage): save_json(RESULTS/'stage2_status.json',dict(stage=stage,time=time.time(),counts=COUNTS,gpus=args.gpus))
    try:
        status('search_and_recheck');winners=search(gpus=args.gpus)
        status('freeze');freeze(winners)
        status('official_asset_preparation');prepare_official()
        status('final');rows=run_queue(formal_jobs(winners),gpus=args.gpus)
        assert len(rows)==40 and all(r['status']=='completed' for r in rows)
        save_json(RESULTS/'final_results.json',rows);write_csv(RESULTS/'final_results.csv',trial_table(rows))
        status('report');audit=full_audit();generate_report(rows,audit)
        save_json(RESULTS/'completion.json',dict(status='completed',**COUNTS,audit=audit))
        status('completed')
    except Exception as error:
        save_json(RESULTS/'stage2_failure.json',dict(status='failed',error=str(error),traceback=traceback.format_exc()))
        status('failed');raise
    finally: lock.close()
    print('Exp10 complete: 184 full trainings; exp10/results/final_report.md',flush=True)
if __name__=='__main__': main()
