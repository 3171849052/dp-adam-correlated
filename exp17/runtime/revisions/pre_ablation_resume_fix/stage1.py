"""Freeze assets/protocol, run independent tests, Monte Carlo and real smoke."""
import fcntl
import importlib.metadata
import subprocess
import sys
from exp17.common import BASE, ROOT, RESULTS, PYTHON, PROTOCOL, METHODS, save, read, sha, code_hashes, assets
from exp17.privacy import calibrate
from exp17.scheduler import run_phase


def run():
    frozen=dict(protocol=PROTOCOL,code=code_hashes(),assets=assets(),
        calibration={s:calibrate(s) for s in ('search','final')},
        environment={p:importlib.metadata.version(p) for p in ('torch','torchvision','opacus','timm','numpy','scipy','safetensors')},
        python=sys.version,executable=sys.executable,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv'],text=True))
    path=RESULTS/'protocol_freeze.json'
    if path.exists() and read(path)!=frozen: raise RuntimeError('Freeze exists with different code/protocol; preserve prior evidence before revalidation')
    save(path,frozen)
    with (RESULTS/'unit_tests.log').open('w') as log:
        subprocess.run([PYTHON,'-B','-m','unittest','discover','-s','exp17/tests','-v'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    from exp17.monte_carlo import run as mc
    measured=mc()
    print('Monte Carlo measured scaling:',measured['scaling'],flush=True)
    trials=[dict(method=m,stage='smoke',lr=.0001,seed=20261026,horizon=h,**({} if m=='adam' else dict(r=.1))) for h in ('search','final') for m in METHODS]
    results=run_phase('smoke',trials)
    assert len(results)==8 and all(r['steps']==2 for r in results)
    save(RESULTS/'acceptance.json',dict(status='passed',freeze_sha256=sha(path),
        tests_sha256=sha(RESULTS/'unit_tests.log'),monte_carlo_sha256=sha(RESULTS/'monte_carlo.json'),
        smoke_results_sha256=sha(RESULTS/'smoke_results.json'),real_smoke_runs=len(results)))
    print('Acceptance passed; ready for full background pipeline.',flush=True)

if __name__=='__main__':
    with (BASE/'runtime/scheduler.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
