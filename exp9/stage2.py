"""Explicit opt-in complete 643-training pipeline; never called by Stage 1."""
import argparse
import json
from exp9 import RESULTS
from exp9.config import COUNTS, save_json

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3],choices=range(4))
    p.add_argument('--cv-per-gpu',type=int,default=1,choices=(1,))
    p.add_argument('--nlp-per-gpu',type=int,default=2,choices=(1,2))
    p.add_argument('--plan',action='store_true',help='Validate and print the fixed plan without starting any training')
    return p

def main():
    args=parser().parse_args();assert len(args.gpus)==len(set(args.gpus))
    from exp9.grid import all_grid
    assert len(all_grid())==COUNTS['search']
    plan=dict(counts=COUNTS,gpus=args.gpus,cv_per_gpu=args.cv_per_gpu,nlp_per_gpu=args.nlp_per_gpu,
              cv_search=177,nlp_search=144,epochs_per_trial=5,
              stage_order=['search','recheck','freeze','final','sweep','report'],
              search_seed=20261101,recheck_seeds=[20261102,20261103],formal_seeds=list(range(20261111,20261121)),
              sweep_epsilons=[2,4,16],sweep_seeds=list(range(20261111,20261114)),
              sweep_protocol='fixed-hyperparameter; only recalibrate DP noise')
    if args.plan: print(json.dumps(plan,indent=2));return
    from exp9.audit import verify_stage1, full_audit
    from exp9.stage1 import unit_tests
    from exp9.grid import write_grid
    from exp9.search import search
    from exp9.frozen import freeze
    from exp9.data import prepare_official
    from exp9.final import run_formal
    from exp9.report import generate_report
    verify_stage1();unit_tests('stage2_');write_grid()
    save_json(RESULTS/'stage2_plan.json',plan)
    def status(stage):
        import time
        save_json(RESULTS/'stage2_status.json',dict(stage=stage,time=time.time(),counts=COUNTS))
    schedule=dict(gpus=args.gpus,cv_per_gpu=args.cv_per_gpu,nlp_per_gpu=args.nlp_per_gpu)
    try:
        status('search_and_recheck');winners=search(**schedule)
        status('freeze');freeze(winners)
        status('official_asset_preparation');prepare_official()
        status('final');run_formal('final',**schedule)
        status('sweep');run_formal('sweep',**schedule)
        status('report');generate_report();status('completed')
    except Exception as error:
        import traceback
        save_json(RESULTS/'stage2_failure.json',dict(status='failed',error=str(error),traceback=traceback.format_exc()))
        raise
    print('Exp9 complete: 643 planned full trainings; report exp9/results/final_report.md',flush=True)

if __name__=='__main__': main()
