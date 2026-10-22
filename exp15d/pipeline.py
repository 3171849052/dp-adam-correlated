"""Durable four-GPU pipeline with an exclusive coordinator lock."""
import fcntl
import os
import time
import traceback
import json
from exp15d import RESULTS
from exp15d.config import save_json


def status(stage,**extra):
    row=dict(stage=stage,pid=os.getpid(),updated=time.time(),gpus=[0,1,2,3],**extra)
    save_json(RESULTS/'pipeline_status.json',row);print(json.dumps(row),flush=True)


def progress(batch,tasks,total):
    path=RESULTS/'pipeline_status.json'
    row=json.loads(path.read_text()) if path.exists() else {}
    row.update(pid=os.getpid(),stage=batch,updated=time.time(),gpus=[0,1,2,3],total=total,
        completed=sum(v['status']=='completed' for v in tasks.values()),invalid=sum(v['status']=='non_finite' for v in tasks.values()),
        active=[dict(trial_id=k,**v) for k,v in tasks.items() if v['status']=='running'])
    save_json(path,row)


def main():
    from exp15d.search import freeze,search_and_review
    from exp15d.scheduler import run_batch
    with (RESULTS/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            freeze();status('eps_screening')
            finals=search_and_review();status('formal',total=len(finals))
            rows=run_batch('formal',finals);save_json(RESULTS/'formal_results.json',rows)
            status('report')
            from exp15d.report import main as report
            report();status('completed')
        except BaseException as error:
            status('failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc());raise


if __name__=='__main__': main()
