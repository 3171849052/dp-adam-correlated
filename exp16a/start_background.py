"""Launch a detached, persistent, single scheduler after phase-one acceptance."""
import fcntl,os,shlex,subprocess,sys,time
from exp16a import ROOT,RESULTS
from exp16a.config import save_json
from exp16a.stage1 import verify_acceptance

def main():
    verify_acceptance()
    lock=open(RESULTS/'pipeline.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    argv=[sys.executable,'-B','-m','exp16a.run_locked','--lock-fd',str(lock.fileno())]
    with (RESULTS/'pipeline.log').open('a') as log:
        p=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
    save_json(RESULTS/'background.json',dict(pid=p.pid,session_id=p.pid,command=shlex.join(argv),launch_command=f'{sys.executable} -B -m exp16a.start_background',log=str(RESULTS/'pipeline.log'),started=time.time()))
    print(f'Launched exp16a pipeline PID {p.pid}',flush=True)
if __name__=='__main__':main()
