"""Detached full Exp18f lifecycle, with a single-controller lock and persistent state."""
import fcntl
import os
import traceback
from exp18f.common import BASE,ROOT,read,save,now,verify_seal,configs,outdir,terminal

def state(status,stage,**extra):
    save(BASE/'pipeline_status.json',dict(status=status,stage=stage,pid=os.getpid(),time=now(),**extra))

def main():
    assert '/envs/curve/' in __import__('sys').executable
    with (BASE/'logs/pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_seal();save(BASE/'logs/pipeline_pid.json',dict(pid=os.getpid(),time=now()))
        from exp18f.report import report
        from exp18f.search import run_search
        stage='axis'
        try:
            state('running',stage);report(plots=False)
            frozen=run_search()
            stage='report';state('running',stage)
            for path in sorted((BASE/'plans').glob('*.json')):
                for cfg in configs(path.stem):assert terminal(cfg,verify=True) is not None
            snapshot={str(p.relative_to(ROOT/'exp18e')):[p.stat().st_size,p.stat().st_mtime_ns]
                for p in (ROOT/'exp18e').rglob('*') if p.is_file()}
            assert snapshot==read(BASE/'exp18e_snapshot.json'),'Exp18e changed'
            state('complete' if frozen['frozen'] else 'complete_no_eligible_freeze','finished')
            counts=report(plots=True)
            save(BASE/'completion_audit.json',dict(time=now(),counts=counts,artifact_hashes_verified=True,
                exp18e_unchanged=True,official_validation_used=False,frozen=frozen['frozen']))
            print('PIPELINE COMPLETE',counts,flush=True)
        except BaseException as error:
            latest=[read(p) for p in (BASE/'logs').glob('*_status.json')]
            failures=[s for s in latest if s.get('status')=='infrastructure_failed']
            state('infrastructure_failed',failures[-1].get('stage',stage) if failures else stage,
                reason=str(error),traceback=traceback.format_exc())
            report(plots=False)
            raise

if __name__=='__main__':main()
