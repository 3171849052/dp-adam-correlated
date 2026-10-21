"""Resume completed jobs; search -> review -> ten-seed formal -> report."""
import fcntl
import os
import time
import traceback
import json
from exp15c import RESULTS
from exp15c.config import save_json
from exp15c.search import require_phase1,freeze,search_and_review
from exp15c.scheduler import run_batch
from exp15c.support import protected_snapshot,read


def status(stage,**extra):
    row=dict(stage=stage,pid=os.getpid(),updated=time.time(),gpus=[1,2,3],**extra)
    save_json(RESULTS/'pipeline_status.json',row);print(json.dumps(row),flush=True)


def main():
    with (RESULTS/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            require_phase1();data=freeze()
            status('search',counts=data['counts'])
            finals=search_and_review()
            status('formal',deduplicated_formal_count=len(finals))
            rows=run_batch('formal',finals)
            save_json(RESULTS/'formal_results.json',rows)
            status('report')
            from exp15c.report import main as report
            report()
            if protected_snapshot()!=read(RESULTS/'protected_before.json'): raise ValueError('Protected history changed')
            status('completed')
        except BaseException as error:
            status('failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
