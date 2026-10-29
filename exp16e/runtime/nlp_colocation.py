"""Authorized scheduling-only entry point; preserves the sealed training source."""
import os,sys,json,time,fcntl,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from exp16e import RESULTS
from exp16e.config import Trial,file_hash,save_json
from exp16e.support import read,code_hashes
from exp16e import scheduler
SCRIPT=Path(__file__).resolve()

def capacity(task,existing):
    return not existing or task=='nlp' and existing==['nlp']

def install():
    scheduler.capacity=capacity
    assert scheduler.GPUS==(1,2,3)
    assert capacity('nlp',['nlp']) and not capacity('nlp',['nlp','nlp'])
    assert not capacity('cv',['nlp']) and not capacity('nlp',['cv'])

def smoke():
    install()
    original=scheduler.gpu_processes
    def only_gpu3():
        occupied=original()
        occupied[1].add(-1);occupied[2].add(-1)
        return occupied
    scheduler.gpu_processes=only_gpu3
    jobs=[Trial('nlp','stpex-mf-momentum-bias-b2lt',.15,20,.001,stage='smoke',replicate=4),Trial('nlp','adam3-mf-momentum-b2lt',.001,10,stage='smoke',replicate=4)]
    rows=scheduler.run_batch('nlp_colocation_smoke',jobs)
    state=read(RESULTS/'nlp_colocation_smoke_status.json');runs=list(state.values())
    assert all(r['physical_gpu']==3 for r in rows)
    assert max(r['started'] for r in runs)<min(r['ended'] for r in runs)
    save_json(RESULTS/'nlp_colocation_acceptance.json',dict(status='passed',script_sha256=file_hash(SCRIPT),code_sha256=code_hashes(),same_gpu=3,overlapping_workers=2,cv_per_gpu=1,nlp_per_gpu=2,nlp_global_max=2,results=rows))

def launch():
    assert '/envs/curve/' in sys.executable
    a=read(RESULTS/'nlp_colocation_acceptance.json')
    assert a['status']=='passed' and a['script_sha256']==file_hash(SCRIPT) and a['code_sha256']==code_hashes()
    lock=(RESULTS/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (RESULTS/'pipeline.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-B',str(SCRIPT),'run'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
    save_json(RESULTS/'background.json',dict(pid=p.pid,started=time.time(),session_detached=True,gpus=[1,2,3],worker_device='cuda:0',cv_max_workers_per_gpu=1,nlp_max_workers_per_gpu=2,nlp_max_workers=2,python=sys.executable,resume='bash exp16e/results/resume.sh',log=str(RESULTS/'pipeline.log'),scheduler_policy_script=str(SCRIPT.relative_to(ROOT)),scheduler_policy_sha256=file_hash(SCRIPT)))
    (RESULTS/'pipeline.pid').write_text(str(p.pid)+'\n');print(p.pid)

def run():
    a=read(RESULTS/'nlp_colocation_acceptance.json')
    assert a['script_sha256']==file_hash(SCRIPT) and a['code_sha256']==code_hashes()
    install()
    from exp16e.pipeline import main
    main()

if __name__=='__main__':
    {'smoke':smoke,'launch':launch,'run':run}[sys.argv[1]]()
