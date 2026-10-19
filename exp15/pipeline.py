"""Run phase two only after phase-one acceptance; resume unchanged successes."""
import fcntl
import json
import os
import time
import traceback
from exp15 import RESULTS
from exp15.config import Trial, save_json, file_hash
from exp15.search import FROZEN, pilot, make_proposal, freeze, load_frozen, search_and_select, require_phase1
from exp15.scheduler import run_batch


def status(stage,**extra):
    row=dict(stage=stage,pid=os.getpid(),updated=time.time(),**extra)
    save_json(RESULTS/'pipeline_status.json',row);print(json.dumps(row),flush=True)


def main():
    with (RESULTS/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            require_phase1()
            if not FROZEN.exists():
                status('pilot');rows=pilot()
                proposal=make_proposal(rows);save_json(RESULTS/'grid_proposal.json',proposal)
                status('freeze');freeze(proposal)
            load_frozen();status('search',grid_sha256=file_hash(FROZEN))
            finals=search_and_select()
            status('formal',deduplicated_formal_count=len(finals))
            rows=run_batch('formal',finals);save_json(RESULTS/'formal_results.json',rows)
            status('report')
            from exp15.report import main as report
            report();status('completed')
        except BaseException as error:
            status('failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
            raise


if __name__=='__main__':main()
