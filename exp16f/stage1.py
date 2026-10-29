"""Unit, matrix, provenance and actual GPU concurrency acceptance."""
import subprocess,sys
from dataclasses import replace
from collections import Counter
from exp16f import ROOT,RESULTS
from exp16f.config import *
from exp16f.support import read,code_hashes
from exp16f.search import centers
from exp16f.scheduler import run_batch

def main():
    assert '/envs/curve/' in sys.executable
    from exp16f.assets import prepare
    prepare()
    with (RESULTS/'unit_tests.log').open('w') as log:
        subprocess.run([sys.executable,'-B','-m','pytest','exp16f/tests','-q','-p','no:cacheprovider','--basetemp=exp16f/runtime/pytest','--junitxml=exp16f/results/tests.xml'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Unit/formula tests passed',flush=True)
    from exp16f.matrices import prepare as matrices
    matrices()
    from exp16f.history import collect
    collect();print('Matrix/privacy/history audit passed',flush=True)
    cv=[replace(c,stage='smoke') for c in centers() if c.task=='cv']
    cv.append(replace(cv[-1],smoke_phase='final'))
    rows=run_batch('cv_smoke',cv)
    nlp=[replace(c,stage='smoke') for c in centers() if c.task=='nlp']
    rows+=run_batch('nlp_concurrent_smoke',nlp)
    runs=list(read(RESULTS/'nlp_concurrent_smoke_status.json').values())
    assert Counter(r['gpu'] for r in runs)=={1:2,2:2,3:2}
    assert max(r['started'] for r in runs)<min(r['ended'] for r in runs)
    repeated=run_batch('repeat_smoke',[replace(cv[-1],replicate=1),replace(nlp[-1],replicate=1)])
    import torch
    for a,b in zip((rows[len(cv)-1],rows[-1]),repeated):
        for k in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace','accuracy','validation_loss','noise_rng_end'):assert a[k]==b[k],k
        from exp16f.history import config
        x=torch.load(config(a).output/'checkpoint.pt',weights_only=True);y=torch.load(config(b).output/'checkpoint.pt',weights_only=True)
        assert all(torch.equal(x['model'][k],y['model'][k]) for k in x['model'])
    save_json(RESULTS/'smoke_results.json',rows+repeated)
    save_json(RESULTS/'acceptance.json',dict(status='passed',code_sha256=code_hashes(),smoke_count=len(rows)+len(repeated),gpus=[1,2,3],nlp_overlapping_workers=6,nlp_per_gpu=2,cv_per_gpu=1))
    print('Acceptance passed',flush=True)
if __name__=='__main__':main()
