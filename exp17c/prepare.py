"""Freeze local sources/assets/calibration; test, then real two-step GPU acceptance."""
import fcntl
import importlib.metadata
import subprocess
import sys
from exp17c.common import BASE, ROOT, RESULTS, PYTHON, PROTOCOL, LR, R, save, read, sha, code_hashes, assets, output
from exp17c.privacy import calibrate
from exp17c.scheduler import run_phase


def prepare():
    calibration={s:calibrate(s) for s in ['search','final']}
    historical=read(ROOT/'exp17b/results/protocol_freeze.json')
    assert calibration==historical['calibration']
    freeze=dict(protocol=PROTOCOL,code=code_hashes(),assets=assets(),calibration=calibration,
        executable=sys.executable,python=sys.version,
        environment={p:importlib.metadata.version(p) for p in ['numpy','scipy','torch','torchvision','opacus','timm','safetensors']},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv'],text=True))
    path=RESULTS/'protocol_freeze.json'
    if path.exists(): assert read(path)==freeze, 'Existing freeze differs; preserve prior evidence before a new preparation'
    save(path,freeze)
    with (RESULTS/'unit_tests.log').open('w') as log:
        subprocess.run([PYTHON,'-B','-m','pytest','exp17c/tests','-q','-o',f'cache_dir={BASE}/runtime/pytest_cache','--basetemp',str(BASE/'runtime/pytest_tmp')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    smoke=[dict(stage='smoke',method='lc',alpha=a,r=0.,lr=LR,seed=1703000+i,steps=2,horizon='search')
        for i,a in enumerate([0.,1e-14,1e-10])]
    results=run_phase('smoke',smoke)
    # Simulate interruption after the last durable step but before the completion summary.
    resume_cfg=smoke[2];folder=output(resume_cfg)
    original=read(folder/'summary.json')
    (folder/'summary.json').rename(folder/'pre_resume_summary.json')
    resumed=run_phase('resume_smoke',[resume_cfg])[0]
    for key in ['accuracy','validation_loss','actual_epsilon','steps']:
        assert original[key]==resumed[key]
    assert original['hashes']['checkpoint.pt']==resumed['hashes']['checkpoint.pt']
    save(RESULTS/'resume_acceptance.json',dict(status='passed',config=resume_cfg,
         checkpoint_unchanged=True,metrics_identical=True))
    save(RESULTS/'acceptance.json',dict(freeze_sha256=sha(path),unit_tests_sha256=sha(RESULTS/'unit_tests.log'),
                                     smoke=[s['config'] for s in results],status='passed'))
    print('Acceptance passed',flush=True)

if __name__=='__main__':
    with (BASE/'runtime/scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare()
