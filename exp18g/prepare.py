"""Freeze the only five formal runs and verify existing local inputs once."""
import csv
import subprocess
import sys
import numpy as np
import torch
from exp18g.common import *
from exp18g.matrices import audit

def prepare():
    assert '/envs/curve/' in sys.executable
    assert not (BASE/'protocol.json').exists()
    audit()
    formal=[dict(stage='formal',method=METHOD,seed=s,steps=T,**PARAMS) for s in SEEDS]
    smoke=[dict(stage='smoke',method=METHOD,seed=20261111,steps=3,gpu=g,**PARAMS) for g in range(4)]
    save(BASE/'plans/formal.json',formal);save(BASE/'plans/smoke.json',smoke)
    save(BASE/'hyperparameters.json',dict(frozen=True,search=False,method=METHOD,**PARAMS))
    protocol=read(ROOT/'exp18f/final_official/protocol.json')
    for k in ('stpex2','stpex2_parameters','source_protocol_sha256'):protocol.pop(k,None)
    protocol.update(experiment='Exp18g',formal_trainings=5,methods={METHOD:PARAMS},
        optimizer='Exact Exp16a LogicalBatch and Adam scale; no adaptive alpha',
        B2LT=dict(T=T,K=K,P=62,Q=5,source='exp18d/matrices/sgd.npz',workload='sgd'),
        gpu='Four UUID-isolated GPUs, one process per GPU, max 4 total, no DDP',
        smoke='Four GPUs, one 3-step STP-B2LT worker each, batch1000, length128, internal64 only',
        historical_results='Reuse all six official algorithms from exp18f/final_official/per_seed.csv',
        statistics='Five paired seeds; ddof=1 sample SD; Student-t(df=4) 95% CI; six paired STP differences',
        source_protocol='exp18f/final_official/protocol.json',
        stp_scale='s_t = 1/(sqrt(vhat_(t-1)) + eps_scale); vhat_0=0; frozen within logical batch',
        stp_update='clip each s*g; sum; add SGD-B2LT noise; divide by s and batch1000; torch.optim.Adam')
    save(BASE/'protocol.json',protocol)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    records=[r.split(',') for r in inventory.splitlines()]
    uuids={r[0].strip():r[1].strip() for r in records}
    assert set(uuids)=={'0','1','2','3'}
    save(BASE/'scheduling.json',dict(physical_gpus=[0,1,2,3],gpu_uuids=uuids,inventory=inventory,
        max_per_gpu=1,max_processes=4,worker_reservation_mib=6500,headroom_mib=700))
    # The already audited raw/tokenized assets are identical to Exp18f's manifest.
    old=read(ROOT/'exp18f/final_official/input_manifest.json')
    assets=[p for p in old if p.startswith(('cache/bert-tiny/','data/sst2/','exp18e/assets/'))]
    for p in assets:assert sha(ROOT/p)==old[p]
    with np.load(ROOT/'exp18e/assets/nlp_split.npz') as s:
        assert len(s['train'])==62000 and len(s['search_validation'])==5349
        assert len(np.intersect1d(s['train'],s['search_validation']))==0
    data_audit=read(ROOT/'exp18f/final_official/data_audit.json')
    assert data_audit['passed'] and data_audit['local_raw_retokenization_equal']
    save(BASE/'acceptance/data_audit.json',dict(passed=True,source='exp18f/final_official/data_audit.json',
        same_asset_hashes=True,train=62000,internal_validation=5349,official_validation=872,downloads=False))
    save(BASE/'acceptance/privacy_audit.json',dict(passed=True,per_seed={str(c['seed']):calibration(c) for c in formal},
        composition='Five runs are not jointly epsilon=8',sampling_amplification=False))
    paths=[ROOT/p for p in assets]
    paths += list((ROOT/'exp18d/matrices').glob('sgd.*'))
    paths += [ROOT/'exp18f/final_official'/n for n in ('per_seed.csv','summary.csv','protocol.json','hyperparameters.json','completion_verification.json','data_audit.json')]
    rows=list(csv.DictReader((ROOT/'exp18f/final_official/per_seed.csv').open()))
    assert len(rows)==30
    for r in rows:
        out=ROOT/r['directory']
        result=read(out/'summary.json')
        assert result['formal'] and result['steps']==310 and result['evaluation']=='official_validation'
        assert result['metrics']['accuracy']==float(r['accuracy']) and result['metrics']['loss']==float(r['loss'])
        paths += [out/n for n in ('summary.json','pairing.json','complete.json')]
    # Record original implementations, then use copies entirely within Exp18g.
    paths += [ROOT/p for p in ('exp16a/optimizer.py','exp16a/clipping.py','exp16e/optimizer.py','exp16e/clipping.py','exp18f/final_official/train.py','exp18f/final_official/model.py','exp18f/final_official/noise.py')]
    save(BASE/'input_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))})
    save(BASE/'source_provenance.json',dict(optimizer='exp16a/optimizer.py, identical Adam path in exp16e',
        clipping='byte-identical exp16a/clipping.py',model='exp18f/final_official/model.py',
        noise='exp18f/final_official/noise.py',training='exp18f/final_official/train.py, STP logical batch substituted',
        scheduler='exp18f/final_official/scheduler.py, 4 GPUs x 1 worker'))
    seal_code();print('PREPARE PASS',flush=True)

if __name__=='__main__':prepare()
