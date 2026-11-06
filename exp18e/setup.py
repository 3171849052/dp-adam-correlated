"""Offline preparation: copied audited matrices and independently checked local SST-2 tokens."""
import shutil
import subprocess
import sys
import numpy as np
import torch
from exp18e.common import (BASE,ROOT,SEEDS,FIXED,HISTORICAL,GROUPS,WORKLOADS,
    read,save,sha,now,load_matrix,calibration,make_config)
from exp18e.matrices import audit,workload


def prepare():
    assert '/envs/curve/' in sys.executable
    assert not (BASE/'acceptance.json').exists()
    torch.set_num_threads(2)
    source=[]
    for folder in ('exp18','exp18b','exp18c','exp18d'):
        source += list((ROOT/folder).glob('*.py'))
        source += list((ROOT/folder).glob('*.json'))
    source += list((ROOT/'exp18d/matrices').glob('*'))+list((ROOT/'exp18d/assets').glob('*'))
    source += list((ROOT/'cache/bert-tiny').glob('*'))
    source += [ROOT/'data/sst2'/f'{s}.parquet' for s in ('train','validation')]
    save(BASE/'upstream_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in source if p.is_file()})
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizer
    tokenizer=BertTokenizer.from_pretrained(str(ROOT/'cache/bert-tiny'),local_files_only=True)
    for split,filename in (('train','nlp_tokens_128.pt'),('validation','nlp_official_tokens_128.pt')):
        frame=pd.read_parquet(ROOT/'data/sst2'/f'{split}.parquet')
        assert len(frame)==(67349 if split=='train' else 872)
        tokens=dict(tokenizer(frame.sentence.tolist(),padding='max_length',truncation=True,
            max_length=128,return_tensors='pt'),labels=torch.tensor(frame.label.tolist()))
        old=torch.load(ROOT/'exp18d/assets'/filename,weights_only=True)
        assert tokens.keys()==old.keys() and all(torch.equal(v,old[k]) for k,v in tokens.items())
        shutil.copyfile(ROOT/'exp18d/assets'/filename,BASE/'assets'/filename)
        if split=='train': raw=frame
    train,valid=train_test_split(np.arange(len(raw)),train_size=62000,random_state=20261008,stratify=raw.label)
    train=np.random.default_rng(20261009).permutation(train).astype(np.int64); valid=np.sort(valid).astype(np.int64)
    with np.load(ROOT/'exp18d/assets/nlp_split.npz') as old:
        np.testing.assert_array_equal(train,old['train']);np.testing.assert_array_equal(valid,old['search_validation'])
    shutil.copyfile(ROOT/'exp18d/assets/nlp_split.npz',BASE/'assets/nlp_split.npz')
    save(BASE/'assets.json',{str(p.relative_to(ROOT)):sha(p) for p in (BASE/'assets').glob('*')})
    save(BASE/'data_audit.json',dict(passed=True,downloads=False,raw_records=[67349,872],
        train=62000,internal_validation=5349,official_validation=872,
        tokens_recomputed_from_local_raw=True,tokens_and_split_equal_exp18d=True,python=sys.executable,time=now()))
    matrices={}
    for name in WORKLOADS.values():
        m=load_matrix(name);meta=m['metadata']
        assert (meta['T'],meta['K'],meta['P'],meta['Q'])==(310,62,62,5)
        assert meta['first_order_pass'] and meta['selected']=='b2lt'
        np.testing.assert_allclose(m['A'],workload(name),atol=1e-12)
        matrices[name]=dict(source=f'exp18d/matrices/{name}.npz',sha256=meta['sha256'],
                            audit=audit(m['A'],m['C'],m['D']))
    save(BASE/'matrix_audit.json',dict(passed=True,matrices=matrices))
    methods={m:calibration(make_config('smoke',m,SEEDS[0],FIXED.get(m,HISTORICAL)))
             for names in GROUPS.values() for m in names}
    save(BASE/'privacy_audit.json',dict(passed=True,methods=methods,
        note='STP parameters here are smoke only; each candidate and formal run recalibrates independently.'))
    coarse=dict(lr=[1e-4,3e-4,1e-3,3e-3,1e-2],clip=[3,10,30,100],eps=[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2])
    save(BASE/'search_space.json',dict(coarse=coarse,historical=HISTORICAL,coarse_seed=20261102,
        review_seeds=[20261103,20261104,20261105],fine_max=14,boundary_max=2,
        ranking='final-step internal accuracy mean over both methods/seeds descending, then loss mean ascending',
        fine='From each of top two coarse winners, one coordinate at a time, geometric midpoint to each adjacent observed coordinate; deduplicate; at most 14.',
        boundary='After fine ranking: if best is at current observed min/max, multiply that coordinate by 0.5/2; choose at most two in deterministic lr,clip,eps order.',
        review='Top two after coarse/fine/boundary; final selection uses ONLY review-seed means over both methods.',
        official_validation='No search worker loads official tokens; only formal workers after freeze.'))
    protocol=read(ROOT/'exp18d/protocol.json')
    protocol.update(experiment='Exp18e',seeds=SEEDS,formal_trainings=30,methods=FIXED,
        stpex2_parameters='Pending common NLP search; both STP methods share one frozen triplet',
        gpu='All four physical GPUs, UUID isolation, free-memory-aware scheduler, <=2 workers per GPU, no DDP',
        selection=read(BASE/'search_space.json'),statistics='5 paired seeds; sample SD ddof=1, Student-t 95% CI for accuracy and loss and paired differences',
        historical_results='No historical checkpoints, metrics or seeds used as formal results',
        privacy_scope='epsilon=8,delta=1e-5 per training; composition of search and formal runs is not jointly epsilon=8; raw internal diagnostics are not DP releases')
    protocol['B2LT']['source']='Byte-identical copy of audited Exp18d matrices; independently re-audited'
    save(BASE/'protocol.json',protocol)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True)
    records=[line.split(',') for line in inventory.splitlines()]
    assert len(records)==4
    save(BASE/'scheduling.json',dict(physical_gpus=[int(r[0]) for r in records],
        gpu_uuids={r[0].strip():r[1].strip() for r in records},inventory=inventory,
        max_per_gpu=2,max_processes=8,worker_reservation_mib=5200,headroom_mib=700,
        scheduler='First fill all four GPUs, then memory-admitted second workers; measured smoke peaks increase reservations if needed.'))
    from exp18e.search import coarse_candidates,stage_plan
    candidates=coarse_candidates();stage_plan('coarse',candidates,[20261102])
    save(BASE/'formal_template.json',dict(seeds=SEEDS,count=30,methods={**FIXED,
        **{m:'shared selection_freeze.json parameters' for m in GROUPS['stpex2']}}))
    save(BASE/'preflight.json',dict(passed=True,time=now(),python=sys.executable,
        coarse_candidates=len(candidates),coarse_trainings=2*len(candidates),formal_trainings=30))
    print('Offline data, matrix audit, per-clip noise calibration, four UUIDs and 242 coarse jobs prepared',flush=True)

if __name__=='__main__':prepare()
