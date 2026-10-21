"""Durable extension -> review -> paired formal training -> audited reporting."""
import fcntl
import os
import time
import traceback
import json
from exp15b import RESULTS
from exp15b.config import save_json, file_hash
from exp15b.search import require_phase1, freeze, search_and_review, FROZEN
from exp15b.scheduler import run_batch
from exp15b.audit import reuse


def status(stage,**extra):
    row=dict(stage=stage,pid=os.getpid(),updated=time.time(),gpus=[0,1,2],reserved_gpu=3,**extra)
    save_json(RESULTS/'pipeline_status.json',row)
    print(json.dumps(row),flush=True)


def main():
    with (RESULTS/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            require_phase1();data=freeze()
            status('search_and_review',grid_sha256=file_hash(FROZEN),counts=data['counts'])
            finals=search_and_review()
            status('formal',deduplicated_formal_count=len(finals))
            for c in finals:reuse(c)
            rows=run_batch('formal',finals)
            save_json(RESULTS/'formal_results.json',rows)
            status('report')
            from exp15b.report import main as report
            report();status('completed')
        except BaseException as error:
            status('failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
            raise


if __name__=='__main__':main()
