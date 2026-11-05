import subprocess,os,fcntl
from exp18 import RESULTS,ROOT
from exp18.common import PYTHON,save,verify_seal,env

def main():
    verify_seal()
    with (RESULTS/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    command=[PYTHON,'-B','-m','exp18.pipeline'];log=RESULTS/'pipeline.log'
    with log.open('a') as f:
        p=subprocess.Popen(command,cwd=ROOT,env=env('1,2,3'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(RESULTS/'background.json',dict(pid=p.pid,command=command,cwd=str(ROOT),log=str(log),physical_gpus=[1,2,3],initial_processes_per_gpu=2,resume=PYTHON+' -B -m exp18.start_background'))
    print(p.pid)
if __name__=='__main__':main()
