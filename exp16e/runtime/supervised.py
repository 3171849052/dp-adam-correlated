"""Resume sealed trials; verify prebuilt official inputs and record native exits."""
import os,sys,subprocess,fcntl,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.update(ARROW_DEFAULT_MEMORY_POOL='system',MALLOC_CONF='background_thread:false',PYTHONFAULTHANDLER='1')
from exp16e import RESULTS
from exp16e.config import save_json,file_hash
from exp16e.support import read,code_hashes
from exp16e.runtime import nlp_six
SCRIPT=Path(__file__).resolve()

def verify_official(official=False):
    assert official,'This continuation only consumes the frozen official input.'
    marker=read(RESULTS/'official_local_assets.json')
    assert marker['historical_tensor_equal']
    for name,digest in marker['files'].items():assert file_hash(ROOT/name)==digest

def check():
    assert '/envs/curve/' in sys.executable
    a=read(RESULTS/'nlp_six_acceptance.json')
    assert a['status']=='passed' and a['script_sha256']==file_hash(nlp_six.SCRIPT) and a['code_sha256']==code_hashes()
    verify_official(True)
    assert read(RESULTS/'final_freeze.json')['code_sha256']==code_hashes()

def run():
    check()
    from exp16e import assets
    # The bytes were rebuilt from data/ and checked against the historical tensors.
    # Do not re-import pandas/Arrow in the long-lived coordinator for a cache hit.
    assets.prepare=verify_official
    nlp_six.run()

def supervise():
    check()
    p=subprocess.Popen([sys.executable,'-B','-X','faulthandler',str(SCRIPT),'run'],cwd=ROOT)
    save_json(RESULTS/'background.json',dict(pid=os.getpid(),pipeline_pid=p.pid,started=time.time(),session_detached=True,gpus=[1,2,3],worker_device='cuda:0',cv_max_workers_per_gpu=1,nlp_max_workers_per_gpu=2,nlp_max_workers=6,resume='bash exp16e/results/resume.sh',log=str(RESULTS/'pipeline.log'),scheduler_policy_script=str(nlp_six.SCRIPT.relative_to(ROOT)),scheduler_policy_sha256=file_hash(nlp_six.SCRIPT),supervisor_sha256=file_hash(SCRIPT),arrow_memory_pool='system',automatic_retry=False))
    rc=p.wait();progress=read(RESULTS/'progress.json')
    save_json(RESULTS/'background_exit.json',dict(time=time.time(),pid=os.getpid(),pipeline_pid=p.pid,returncode=rc,signal=-rc if rc<0 else None,last_progress=progress))
    if rc!=0 or progress['phase']!='completed':
        save_json(RESULTS/'progress.json',dict(phase='failed',time=time.time(),pipeline_returncode=rc,signal=-rc if rc<0 else None,last_progress=progress,error='Coordinator exited; see background_exit.json and pipeline.log. No automatic retry.'))
        raise SystemExit(1)

def launch():
    check()
    lock=(RESULTS/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (RESULTS/'pipeline.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-B','-X','faulthandler',str(SCRIPT),'supervise'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
    (RESULTS/'pipeline.pid').write_text(str(p.pid)+'\n');print(p.pid)

if __name__=='__main__':{'run':run,'launch':launch,'supervise':supervise,'check':check}[sys.argv[1]]()
