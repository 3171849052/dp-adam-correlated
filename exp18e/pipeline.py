"""Complete detached search -> freeze -> 30 formal trainings -> statistics."""
import fcntl
import os
from exp18e.common import BASE,verify_seal,save,now


def main():
    with (BASE/'runtime/pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_seal()
        def status(stage,state='running'):
            save(BASE/'logs/pipeline_status.json',dict(pid=os.getpid(),stage=stage,status=state,time=now()))
        status('search')
        from exp18e.search import run_search
        from exp18e.scheduler import run
        from exp18e.report import report
        try:
            run_search();status('formal');report();run('formal')
            status('final_audit');verify_seal();assert report()
        except BaseException:
            status('inspect_logs','failed');raise
        status('finished','complete')
        print('Exp18e COMPLETE: search, common freeze, 30 fresh formal trainings, paired statistics and report',flush=True)

if __name__=='__main__':main()
