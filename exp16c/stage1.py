"""Unit tests, immutable-input audit and real ViT smoke before acceptance."""
import os,subprocess,sys
from pathlib import Path
from exp16c import ROOT,BASE,RESULTS
from exp16c.config import *
from exp16c.support import read,code_hashes

def snapshot():
    return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns] for folder in ('exp16a','exp16b','exp15c','data','cache') for p in sorted((ROOT/folder).rglob('*')) if p.is_file()}
def verify_readonly():
    if snapshot()!=read(RESULTS/'protected_before.json'):raise ValueError('Historical/data/cache files changed')
def verify_acceptance():
    a=read(RESULTS/'phase1_acceptance.json')
    if a['status']!='passed' or a['code_sha256']!=code_hashes():raise ValueError('Acceptance missing or training code changed')
    for name,h in a['evidence_sha256'].items():
        if file_hash(RESULTS/name)!=h:raise ValueError('Acceptance evidence changed')
def run():
    RESULTS.mkdir(exist_ok=True)
    if not (RESULTS/'protected_before.json').exists():save_json(RESULTS/'protected_before.json',snapshot())
    from exp16c.matrices import prepare
    prepare()
    with (RESULTS/'unit_tests.log').open('w') as log:
        result=subprocess.run([sys.executable,'-B','-m','pytest','-q','exp16c/tests','-p','no:cacheprovider',f'--basetemp={BASE}/runtime/pytest',f'--junitxml={RESULTS}/tests.xml'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Unit tests failed; see unit_tests.log')
    from exp16c.scheduler import run_batch
    smoke=[Trial('cv',m,.003 if m in A_METHODS else .1 if m==STPEX else .5,12. if m in A_METHODS else 3.,3. if m==STPEX else None,stage='smoke') for m in (*A_METHODS,STPEX,*MOMENTUM_METHODS)]
    # Also exercise formal horizon without opening official test; duplicate checks RNG parity.
    smoke += [Trial('cv',m,.5,3.,stage='smoke',smoke_phase='final') for m in MOMENTUM_METHODS]
    smoke += [Trial('cv',MOMENTUM_METHODS[0],.5,3.,stage='smoke',replicate=1)]
    runs=run_batch('smoke',smoke)
    for phase in ('search','final'):
        group=[r for r in runs if r['method'] in MOMENTUM_METHODS and r['smoke_phase']==phase]
        for r in group[1:]:
            for key in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace'):
                if r[key]!=group[0][key]:raise ValueError(f'Smoke randomness mismatch: {key}')
    a=next(r for r in runs if r['method']==MOMENTUM_METHODS[0] and r['replicate']==0 and r['smoke_phase']=='search');b=runs[-1]
    import torch
    x=torch.load(Trial(**{k:a[k] for k in Trial.__dataclass_fields__}).output/'checkpoint.pt',weights_only=True)
    y=torch.load(Trial(**{k:b[k] for k in Trial.__dataclass_fields__}).output/'checkpoint.pt',weights_only=True)
    if not all(torch.equal(x['model'][k],y['model'][k]) for k in x['model']):raise ValueError('Smoke exact replay mismatch')
    save_json(RESULTS/'smoke_results.json',runs);verify_readonly()
    save_json(RESULTS/'phase1_acceptance.json',dict(status='passed',code_sha256=code_hashes(),real_model_smokes=len(runs),exact_replay=True,momentum_randomness_matched=True,evidence_sha256={name:file_hash(RESULTS/name) for name in ('unit_tests.log','tests.xml','smoke_results.json','matrix_audit.json')}))
    print('exp16c acceptance passed',flush=True)
if __name__=='__main__':run()
