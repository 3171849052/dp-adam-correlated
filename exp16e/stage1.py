"""Acceptance gate: unit/formula/privacy tests, historical audit, real GPU prefixes."""
import subprocess,sys
from dataclasses import replace
from exp16e import ROOT,RESULTS
from exp16e.config import *
from exp16e.support import read,code_hashes
from exp16e.assets import prepare
from exp16e.scheduler import run_batch

def main():
    assert '/envs/curve/' in sys.executable
    prepare()
    with (RESULTS/'unit_tests.log').open('w') as log:
        subprocess.run([sys.executable,'-B','-m','pytest','exp16e/tests','-q','-p','no:cacheprovider','--basetemp=exp16e/runtime/pytest','--junitxml=exp16e/results/tests.xml'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    from exp16e.matrices import prepare as matrices
    matrices()
    from exp16e.history import collect,incumbents,config,formal
    rows=collect('search')+collect('review')
    assert sum(r['task']=='nlp' and r['method']==STPEX and r['source'].startswith('exp16b/') for r in rows)==27
    save_json(RESULTS/'historical_internal_results.json',rows)
    # Verify all unchanged incumbent formal records before scheduling any new work.
    count=0
    for task,methods in incumbents().items():
        for v in methods.values():
            for seed in FINAL_SEEDS:
                assert formal(replace(config(v),stage='final',seed=seed)) is not None;count+=1
    save_json(RESULTS/'history_audit.json',dict(status='passed',internal_count=len(rows),incumbent_formal_count=count))
    smoke=[]
    for task in ('cv','nlp'):
        for method,lr,clip,eps in [(MOMENTUM[0],.3,.3,None),(MOMENTUM[1],.3,20,None),(ADAM3[0],.0003,.3,None),(ADAM3[2],.001,10,None),(STPEX,.15,20,.001 if task=='nlp' else 3),(ADAM[-1],.0163 if task=='nlp' else .003,500 if task=='nlp' else 200,.00164316767252 if task=='nlp' else .05),(ADAM[1],.0163 if task=='nlp' else .003,40 if task=='nlp' else 20,None)]:
            smoke.append(Trial(task,method,lr,clip,eps,stage='smoke',replicate=2))
    smoke.append(replace(smoke[3],smoke_phase='final'))
    # Same seed repeated prefix checks entire checkpoint and non-noise random protocol.
    smoke.extend(replace(c,replicate=3) for c in (smoke[1],smoke[11]))
    results=run_batch('smoke',smoke)
    for original,repeat in ((results[1],results[-2]),(results[11],results[-1])):
        for k in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace','accuracy','validation_loss','noise_rng_end'):
            assert original[k]==repeat[k],k
        # Torch archives may include serialization IDs; compare tensor content.
        import torch
        a=torch.load(Trial(**{k:original[k] for k in Trial.__dataclass_fields__}).output/'checkpoint.pt',weights_only=True)
        b=torch.load(Trial(**{k:repeat[k] for k in Trial.__dataclass_fields__}).output/'checkpoint.pt',weights_only=True)
        assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    save_json(RESULTS/'smoke_results.json',results)
    save_json(RESULTS/'acceptance.json',dict(status='passed',code_sha256=code_hashes(),unit_tests='tests.xml',smoke_count=len(results),gpu_policy=[1,2,3],nlp_concurrency=2,history='history_audit.json',matrix='matrix_audit.json'))
    print('Acceptance passed',flush=True)

if __name__=='__main__':main()
