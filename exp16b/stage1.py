"""Acceptance before any coarse worker, with paired real-model repetitions."""
import os,sys,subprocess,time
from dataclasses import replace
from exp16b import ROOT,BASE,RESULTS
from exp16b.config import Trial,METHODS,file_hash,save_json
from exp16b.support import read,code_hashes,platform_hashes,verify_assets
from exp16b.search import freeze

def verify_acceptance():
    a=read(RESULTS/'phase1_acceptance.json')
    assert a['status']=='passed' and a['platform_sha256']==platform_hashes()
    for name,h in a['evidence_sha256'].items():assert file_hash(RESULTS/name)==h
    from exp16b.matrices import load
    for n in (225,310):
        for m in ('bandinvmf','b2lt'):load('momentum-bias',m,n)
    return a

def run():
    import importlib.metadata
    freeze();save_json(RESULTS/'assets.json',verify_assets())
    save_json(RESULTS/'environment.json',dict(python=sys.executable,packages={n:importlib.metadata.version(n) for n in ('torch','numpy','timm','opacus','transformers','torchvision','pytest')},gpu=subprocess.check_output(['nvidia-smi'],text=True)))
    from exp16b.matrices import prepare
    prepare()
    args=[sys.executable,'-B','-m','pytest','exp16b/tests','-q','-p','no:cacheprovider','--basetemp=exp16b/runtime/pytest','--junitxml=exp16b/results/tests.xml']
    with (RESULTS/'unit_tests.log').open('w') as f:
        result=subprocess.run(args,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Acceptance tests failed: unit_tests.log')
    from exp16b.scheduler import run_batch
    jobs=[Trial(task,m,.15,3. if task=='cv' else 20.,.1 if task=='cv' else .001,stage='smoke') for task in ('cv','nlp') for m in METHODS]
    jobs += [replace(c,replicate=1) for c in jobs]
    rows=run_batch('smoke',jobs);save_json(RESULTS/'smoke_results.json',rows)
    import torch
    for c in jobs:
        if c.replicate!=1:continue
        other=replace(c,replicate=0)
        a=torch.load(c.output/'checkpoint.pt',map_location='cpu',weights_only=False)
        b=torch.load(other.output/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    for task in ('cv','nlp'):
        rs=[r for r in rows if r['task']==task]
        for key in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace','noise_rng_start','noise_rng_end'):
            assert all(r[key]==rs[0][key] for r in rs)
    evidence={n:file_hash(RESULTS/n) for n in ('unit_tests.log','tests.xml','smoke_results.json','matrix_audit.json','assets.json','coarse_grid.json','environment.json','source_provenance.json')}
    save_json(RESULTS/'phase1_acceptance.json',dict(status='passed',smokes=len(rows),core_sha256=code_hashes(),platform_sha256=platform_hashes(),evidence_sha256=evidence))
    print('Acceptance passed',flush=True)
if __name__=='__main__':run()
