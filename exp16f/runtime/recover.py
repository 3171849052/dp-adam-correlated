"""Selection-only recovery; preserve the sealed training implementation and artifacts."""
import fcntl,os,subprocess,sys,time
from pathlib import Path
from exp16f import ROOT,RESULTS
from exp16f import search
from exp16f.config import save_json,file_hash
from exp16f.support import read,code_hashes
SCRIPT=Path(__file__).resolve()
TEST=ROOT/'exp16f/tests/test_recovery.py'

def candidates(rows,c):
    completed=[]
    for r in rows:
        assert r['stage'] in ('search','review'), 'Official stage cannot enter selection'
        if r['status']=='completed':
            assert r['official_evaluation_used'] is False, 'Official evaluation cannot enter selection'
            completed.append(r)
        else:
            assert r['status']=='non_finite' and r['error'].startswith('FloatingPointError:'), 'Uninspected trial failure'
            assert r.get('official_evaluation_used',False) is False
    return sorted([r for r in completed if r['task']==c.task and r['method']==c.method and r['stage']=='search'],key=search.rank)

def check():
    a=read(RESULTS/'recovery_acceptance.json')
    assert a['status']=='passed' and a['script_sha256']==file_hash(SCRIPT) and a['test_sha256']==file_hash(TEST)
    assert a['code_sha256']==code_hashes()==read(RESULTS/'acceptance.json')['code_sha256']
    return a

def run():
    check();search.candidates=candidates
    from exp16f import report
    original=report.generate
    def generate(formal,freeze):
        original(formal,freeze)
        audit=read(RESULTS/'final_audit.json')
        audit['selection_recovery']=dict(source=str(SCRIPT.relative_to(ROOT)),sha256=file_hash(SCRIPT),acceptance_sha256=file_hash(RESULTS/'recovery_acceptance.json'))
        save_json(RESULTS/'final_audit.json',audit)
    report.generate=generate
    from exp16f.pipeline import main
    main()

def launch():
    check();assert '/envs/curve/' in sys.executable
    lock=(RESULTS/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (RESULTS/'pipeline.log').open('a') as log:
        log.write('\nSelection recovery: exclude sealed numerical failures before evaluation-field checks.\n');log.flush()
        p=subprocess.Popen([sys.executable,'-B','-m','exp16f.runtime.recover','run'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
    save_json(RESULTS/'background.json',dict(pid=p.pid,started=time.time(),session_detached=True,gpus=[1,2,3],worker_device='cuda:0',cv_max_workers_per_gpu=1,nlp_max_workers_per_gpu=2,nlp_max_workers=6,python=sys.executable,resume='bash exp16f/results/resume.sh',log=str(RESULTS/'pipeline.log'),recovery_source=str(SCRIPT.relative_to(ROOT)),recovery_sha256=file_hash(SCRIPT)))
    (RESULTS/'pipeline.pid').write_text(str(p.pid)+'\n');print(p.pid,flush=True)

if __name__=='__main__':
    {'run':run,'launch':launch}[sys.argv[1]]()
