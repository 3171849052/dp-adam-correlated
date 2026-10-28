"""Freeze local sources/assets/calibration; test, then real two-step GPU acceptance."""
import fcntl
import importlib.metadata
import subprocess
import sys
from exp17b.common import BASE, ROOT, RESULTS, PYTHON, PROTOCOL, LR, R, save, read, sha, code_hashes, assets
from exp17b.privacy import calibrate
from exp17b.scheduler import run_phase


def prepare():
    calibration={s:calibrate(s) for s in ['search','final']}
    historical=read(ROOT/'exp17/results/protocol_freeze.json')
    assert calibration==historical['calibration']
    freeze=dict(protocol=PROTOCOL,code=code_hashes(),assets=assets(),calibration=calibration,
        executable=sys.executable,python=sys.version,
        environment={p:importlib.metadata.version(p) for p in ['numpy','scipy','torch','torchvision','opacus','timm','safetensors']},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv'],text=True))
    path=RESULTS/'protocol_freeze.json'
    if path.exists(): assert read(path)==freeze, 'Existing freeze differs; preserve prior evidence before a new preparation'
    save(path,freeze)
    with (RESULTS/'unit_tests.log').open('w') as log:
        subprocess.run([PYTHON,'-B','-m','pytest','exp17b/tests','-q','-o',f'cache_dir={BASE}/runtime/pytest_cache','--basetemp',str(BASE/'runtime/pytest_tmp')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    smoke=[dict(stage='smoke',method=m,alpha=a,r=r,lr=LR,seed=20261028,steps=2,horizon=h)
        for m,a,r,h in [('smooth',0.,R,'search'),('lc',1.,R,'search'),('gh',.5,R,'search'),('gh',1.,0.,'final')]]
    results=run_phase('smoke',smoke)
    save(RESULTS/'acceptance.json',dict(freeze_sha256=sha(path),unit_tests_sha256=sha(RESULTS/'unit_tests.log'),
                                     smoke=[s['config'] for s in results],status='passed'))
    print('Acceptance passed',flush=True)

if __name__=='__main__':
    with (BASE/'runtime/scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare()
