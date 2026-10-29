"""CPU correctness now; full physical-batch GPU acceptance after predecessor."""
import os,sys,subprocess
from exp16d import ROOT,BASE,RESULTS
from exp16d.config import *
from exp16d.support import read,code_hashes

def verify_acceptance(gpu=False):
    a=read(RESULTS/('gpu_acceptance.json' if gpu else 'phase1_acceptance.json'))
    if a['status']!='passed' or a['code_sha256']!=code_hashes():raise ValueError('Acceptance missing or code changed')
    for name,h in a['evidence_sha256'].items():
        if file_hash(RESULTS/name)!=h:raise ValueError('Acceptance evidence changed')
def gpu_acceptance():
    if (RESULTS/'gpu_acceptance.json').exists():verify_acceptance(gpu=True);return
    verify_acceptance()
    from exp16d.scheduler import run_batch
    jobs=[Trial('cv',m,.0004,20,stage='smoke',smoke_phase=p) for p in ('search','final') for m in METHODS]
    rows=run_batch('smoke',jobs)
    for p in ('search','final'):
        group=[r for r in rows if r['smoke_phase']==p]
        for r in group[1:]:
            for key in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace'):
                assert r[key]==group[0][key],key
    save_json(RESULTS/'smoke_results.json',rows)
    save_json(RESULTS/'gpu_acceptance.json',dict(status='passed',code_sha256=code_hashes(),real_model_smokes=8,paired_randomness=True,evidence_sha256={'smoke_results.json':file_hash(RESULTS/'smoke_results.json')}))
def run():
    from exp16d.matrices import prepare
    prepare()
    with (RESULTS/'unit_tests.log').open('w') as log:
        p=subprocess.run([sys.executable,'-B','-m','pytest','-q','exp16d/tests','-p','no:cacheprovider',f'--basetemp={BASE}/runtime/pytest',f'--junitxml={RESULTS}/tests.xml'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('Tests failed; see unit_tests.log')
    save_json(RESULTS/'phase1_acceptance.json',dict(status='passed',code_sha256=code_hashes(),scope='CPU algorithm/privacy/numerics and real pretrained model smoke; GPU full batch gate queued after exp16c',evidence_sha256={n:file_hash(RESULTS/n) for n in ('unit_tests.log','tests.xml','matrix_audit.json','cpu_smoke.json')}))
if __name__=='__main__':run()
