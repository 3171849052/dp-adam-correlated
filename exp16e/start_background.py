"""Detach with inherited flock; no automatic retries of failed/partial trials."""
import fcntl,os,subprocess,sys,time
from exp16e import RESULTS,ROOT
from exp16e.config import save_json
from exp16e.support import read,code_hashes

def main():
    assert '/envs/curve/' in sys.executable,sys.executable
    a=read(RESULTS/'acceptance.json');assert a['status']=='passed' and a['code_sha256']==code_hashes()
    lock=(RESULTS/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (RESULTS/'pipeline.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-B','-m','exp16e.pipeline'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
    save_json(RESULTS/'background.json',dict(pid=p.pid,started=time.time(),session_detached=True,gpus=[1,2,3],worker_device='cuda:0',max_workers_per_gpu=1,nlp_max_workers=2,python=sys.executable,resume=f'{sys.executable} -B -m exp16e.start_background',log=str(RESULTS/'pipeline.log')))
    (RESULTS/'pipeline.pid').write_text(str(p.pid)+'\n');print(p.pid)

if __name__=='__main__':main()
