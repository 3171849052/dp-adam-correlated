"""Prepare, test, execute real model smokes, then freeze algorithm and matrices."""
import argparse,os,subprocess,sys,time
import importlib.metadata
from dataclasses import replace
from exp16a import ROOT,BASE,RESULTS
from exp16a.config import Trial,METHODS,save_json,file_hash
from exp16a.support import read,code_hashes,platform_hashes,protected_snapshot,verify_protected,verify_assets
from exp16a.search import GRIDS,category,freeze_json,coarse_jobs

def matrix_hashes():return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted((RESULTS/'matrices').glob('*'))}
def verify_acceptance():
    p=RESULTS/'phase1_acceptance.json'
    a=read(p)
    if a['status']!='passed' or a['core_sha256']!=code_hashes() or a['platform_sha256']!=platform_hashes() or a['matrix_sha256']!=matrix_hashes():raise ValueError('Phase1 acceptance/frozen core mismatch')
    for name,h in a['evidence_sha256'].items():
        if file_hash(RESULTS/name)!=h:raise ValueError(f'Acceptance evidence changed: {name}')
    return a

def smoke_jobs():
    jobs=[]
    for m in METHODS:
        for task in ('cv','nlp'):
            g=GRIDS[task][category(m)];v={k:values[0] for k,values in g.items()}
            jobs.append(Trial(task,m,**v,stage='smoke',seed=20261099))
    jobs += [replace(c,smoke_phase='final') for c in jobs if c.task=='cv']
    jobs += [replace(c,replicate=1) for c in jobs[:28] if c.method in ('adam','adam-mf-sgd-stp-b2lt')]
    return jobs

def run():
    started=time.time()
    packages={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','timm','opacus','transformers','torchvision','pytest')}
    save_json(RESULTS/'environment.json',dict(python=sys.version,executable=sys.executable,packages=packages,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True),git_commit=__import__('exp16a.support',fromlist=['version']).version(),pdf_sha256=file_hash(ROOT/'stp-ex.pdf')))
    p=RESULTS/'protected_before.json'
    if not p.exists():save_json(p,protected_snapshot())
    verify_protected();assets=verify_assets();save_json(RESULTS/'assets.json',assets)
    from exp16a.matrices import prepare
    prepare()
    protocol=read(BASE/'protocol.json');freeze_json('protocol_freeze.json',protocol)
    freeze_json('coarse_grid.json',dict(trials=[c.values() for c in coarse_jobs()]))
    args=[sys.executable,'-B','-m','pytest','exp16a/tests','-q','-p','no:cacheprovider','--basetemp=exp16a/runtime/pytest_tmp','--junitxml=exp16a/results/tests.xml']
    with (RESULTS/'unit_tests.log').open('w') as f:
        p=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('Unit/matrix tests failed; see unit_tests.log')
    save_json(RESULTS/'test_results.json',dict(status='passed',command=args,core_sha256=code_hashes(),log_sha256=file_hash(RESULTS/'unit_tests.log')))
    from exp16a.scheduler import run_batch
    rows=run_batch('smoke',smoke_jobs());save_json(RESULTS/'smoke_results.json',rows)
    # All methods share initialization, training order, augmentation/dropout for paired seeds.
    for task in ('cv','nlp'):
        for phase in ('search','final') if task=='cv' else ('search',):
            subset=[r for r in rows if r['task']==task and r['smoke_phase']==phase]
            for key in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace'):
                if any(r[key]!=subset[0][key] for r in subset):raise ValueError(f'Unpaired smoke {task}/{phase}/{key}')
    import torch
    for r in rows:
        if r['replicate']!=1:continue
        c=Trial(**{k:r[k] for k in Trial.__dataclass_fields__});original=replace(c,replicate=0)
        one=torch.load(c.output/'checkpoint.pt',map_location='cpu',weights_only=False);two=torch.load(original.output/'checkpoint.pt',map_location='cpu',weights_only=False)
        for name,t in one['model'].items():
            if not torch.equal(t,two['model'][name]):raise ValueError('Smoke repetition differs')
    verify_protected()
    evidence={n:file_hash(RESULTS/n) for n in ('unit_tests.log','tests.xml','test_results.json','smoke_results.json','matrix_audit.json','protocol_freeze.json','assets.json','coarse_grid.json','environment.json','matrix_solver.log','matrix_generation.json','source_provenance.json')}
    save_json(RESULTS/'phase1_acceptance.json',dict(status='passed',smokes=len(rows),core_sha256=code_hashes(),platform_sha256=platform_hashes(),matrix_sha256=matrix_hashes(),evidence_sha256=evidence,seconds=time.time()-started))
    print('Phase 1 passed and core/matrices frozen',flush=True)
if __name__=='__main__':run()
