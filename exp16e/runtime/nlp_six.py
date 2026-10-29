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

def runnable(task,active):
    return task!='nlp' or sum(c.task=='nlp' for c,p,g,log in active.values())<6

def install():
    scheduler.capacity=capacity
    scheduler.runnable=runnable
    assert scheduler.GPUS==(1,2,3)
    assert capacity('nlp',['nlp']) and not capacity('nlp',['nlp','nlp'])
    assert not capacity('cv',['nlp']) and not capacity('nlp',['cv'])
    c=Trial('nlp','adam-iid',.002,.3)
    active={i:(c,None,1+i//2,None) for i in range(6)}
    assert not runnable('nlp',active) and runnable('cv',active)
    active.pop(5);assert runnable('nlp',active)

def smoke():
    install()
    jobs=[Trial('nlp',m,lr,clip,eps,stage='smoke',replicate=5) for m,lr,clip,eps in [
        ('stpex-mf-momentum-bias-b2lt',.15,20,.001),
        ('adam3-mf-momentum-b2lt',.001,10,None),
        ('momentum-mf-sgd-b2lt',.3,20,None),
        ('adam-mf-sgd-stp-b2lt',.0163,500,.00164316767252),
        ('adam3-mf-sgd-b2lt',.001,20,None),
        ('adam-mf-momentum-b2lt',.0163,10,None)]]
    rows=scheduler.run_batch('nlp_six_smoke',jobs)
    state=read(RESULTS/'nlp_six_smoke_status.json');runs=list(state.values())
    from collections import Counter
    assert Counter(r['physical_gpu'] for r in rows)=={1:2,2:2,3:2}
    assert max(r['started'] for r in runs)<min(r['ended'] for r in runs)
    save_json(RESULTS/'nlp_six_acceptance.json',dict(status='passed',script_sha256=file_hash(SCRIPT),code_sha256=code_hashes(),gpus=[1,2,3],overlapping_workers=6,cv_per_gpu=1,nlp_per_gpu=2,nlp_global_max=6,results=rows))

def launch():
    assert '/envs/curve/' in sys.executable
    a=read(RESULTS/'nlp_six_acceptance.json')
    assert a['status']=='passed' and a['script_sha256']==file_hash(SCRIPT) and a['code_sha256']==code_hashes()
    lock=(RESULTS/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (RESULTS/'pipeline.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-B',str(SCRIPT),'run'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
    save_json(RESULTS/'background.json',dict(pid=p.pid,started=time.time(),session_detached=True,gpus=[1,2,3],worker_device='cuda:0',cv_max_workers_per_gpu=1,nlp_max_workers_per_gpu=2,nlp_max_workers=6,python=sys.executable,resume='bash exp16e/results/resume.sh',log=str(RESULTS/'pipeline.log'),scheduler_policy_script=str(SCRIPT.relative_to(ROOT)),scheduler_policy_sha256=file_hash(SCRIPT)))
    (RESULTS/'pipeline.pid').write_text(str(p.pid)+'\n');print(p.pid)

def run():
    a=read(RESULTS/'nlp_six_acceptance.json')
    assert a['script_sha256']==file_hash(SCRIPT) and a['code_sha256']==code_hashes()
    install()
    # Keep the frozen reporting implementation; attach the active scheduling policy.
    from exp16e import report
    original_generate=report.generate
    def generate(formal,freeze):
        original_generate(formal,freeze)
        audit=read(RESULTS/'final_audit.json')
        audit.update(nlp_max_concurrent=6,nlp_max_workers_per_gpu=2,cv_max_workers_per_gpu=1,scheduler_policy_sha256=file_hash(SCRIPT),scheduler_policy_source=str(SCRIPT.relative_to(ROOT)))
        save_json(RESULTS/'final_audit.json',audit)
    report.generate=generate
    from exp16e.pipeline import main
    main()

if __name__=='__main__':
    {'smoke':smoke,'launch':launch,'run':run}[sys.argv[1]]()
