"""Validate the 160-run plan, or execute Stage 1 through Stage 4."""
import argparse
import fcntl
import json
import time
import traceback
from exp10b import RESULTS
from exp10b.config import COUNTS,save_json,immutable_json
from exp10b.protocol import plan,search_jobs,choose_winners,formal_jobs

def execute(gpus):
    from exp10b.audit import verify_stage0,audit_pairing,full_audit
    from exp10b.launcher import run_queue
    from exp10b.frozen import freeze
    from exp10b.data import prepare_official
    from exp10b.report import search_report,final_report
    RESULTS.mkdir(parents=True,exist_ok=True)
    with (RESULTS/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        platform=verify_stage0()
        assert set(gpus)<=set(platform['gpus'])
        # GPU placement is operational; scientific plan is precommitted in Stage 0.
        assert json.loads((RESULTS/'full_plan.json').read_text())==plan(platform['gpus'])
        started=time.time()
        def status(stage,**fields):
            save_json(RESULTS/'experiment_status.json',dict(stage=stage,started_at_unix=started,
                updated_at_unix=time.time(),counts=COUNTS,**fields))
        try:
            status('stage1',status='running')
            first=run_queue(search_jobs('stage1'),gpus=gpus)
            immutable_json(RESULTS/'stage1_results.json',first)
            audit_pairing(first)
            status('stage2',status='running',completed_full_trainings=36)
            second=run_queue(search_jobs('stage2'),gpus=gpus)
            immutable_json(RESULTS/'stage2_results.json',second)
            rows=first+second;audit_pairing(rows)
            status('stage3',status='running',completed_full_trainings=120)
            winners,evidence=choose_winners(rows)
            immutable_json(RESULTS/'selection_evidence.json',evidence)
            winners=freeze(winners)
            search_report(rows,evidence)
            # Official assets are first prepared after immutable selection.
            prepare_official()
            status('stage4',status='running',completed_full_trainings=120,winners=winners)
            final=run_queue(formal_jobs(winners),gpus=gpus)
            immutable_json(RESULTS/'final_results.json',final)
            audit_pairing(final);audit=full_audit()
            final_report(rows,final,evidence,winners,audit)
            status('complete',status='completed',completed_full_trainings=160,winners=winners)
            save_json(RESULTS/'completion.json',dict(status='completed',full_trainings=160,counts=COUNTS))
            print('Exp10b complete: 36 + 84 + 40 = 160 full trainings.',flush=True)
        except Exception as error:
            previous=json.loads((RESULTS/'experiment_status.json').read_text())
            previous.update(status='failed',error=str(error),traceback=traceback.format_exc())
            save_json(RESULTS/'experiment_status.json',previous)
            raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',type=int,nargs='+',choices=(1,2,3),default=[1,2,3])
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();protocol=plan(args.gpus)
    if args.plan:
        print(json.dumps(protocol,indent=2));return
    execute(args.gpus)

if __name__=='__main__': main()
