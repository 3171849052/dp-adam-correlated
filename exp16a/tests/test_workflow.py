from dataclasses import replace
import copy
import numpy as np
import pytest
from exp16a import RESULTS
from exp16a.config import Trial,METHODS,save_json
from exp16a.search import coarse_jobs,fine_jobs,rank,refine,top_two,select,GRIDS,category
from exp16a.scheduler import capacity,verify

def test_counts_and_independent_budgets():
    jobs=coarse_jobs();assert len(jobs)==348 and len({c.id for c in jobs})==348
    for task in ('cv','nlp'):
        for m in METHODS:
            cs=[c for c in jobs if (c.task,c.method)==(task,m)]
            assert len(cs)==dict(adam=3,iid=9,mf=9,stp=27)[category(m)]
    rows=[dict(**c.values(),trial_id=c.id,status='completed',accuracy=.5,validation_loss=1.,official_evaluation_used=False) for c in jobs]
    fine,basis=fine_jobs(rows);assert len(fine)==348
    assert len(top_two(rows)['cv'])==14
    assert len({c.id for c in fine}&{c.id for c in jobs})>=28

def test_refinement_and_loss_ties():
    assert refine([1.,4.,16.],4.)==[2.,4.,8.]
    assert refine([1.,4.,16.],1.)==[.5,1.,2.]
    assert refine([1.,4.,16.],16.)==[8.,16.,32.]
    assert rank([dict(status='completed',accuracy=.6,validation_loss=2.,trial_id='a'),dict(status='completed',accuracy=.6,validation_loss=1.,trial_id='b')])[0]['trial_id']=='b'
    with pytest.raises(ValueError):rank([dict(official_evaluation_used=True)])

def test_slot_rules():
    for task in ('cv','nlp'):assert capacity(task,[])
    assert capacity('nlp',['nlp'])
    for t,e in [('cv',['cv']),('nlp',['cv']),('cv',['nlp']),('nlp',['nlp','nlp'])]:assert not capacity(t,e)

def test_success_reuse_requires_artifacts(tmp_path):
    # Files are under exp16a/runtime via pytest's configured basetemp.
    from exp16a.support import code_hashes
    from exp16a.config import json_hash,file_hash
    c=Trial('cv','adam',.001,stage='smoke',replicate=919)
    c.output.mkdir(parents=True,exist_ok=True)
    from exp16a.support import verify_assets
    resolved=dict(c.protocol(),code_sha256=code_hashes(),input_assets_sha256=verify_assets());h=json_hash(resolved)
    save_json(c.output/'config.json',dict(resolved,config_sha256=h))
    artifact=c.output/'test_artifact.txt';artifact.write_text('original')
    row=dict(**c.values(),trial_id=c.id,status='completed',code_sha256=code_hashes(),config_sha256=h,optimizer_steps=2,
        accuracy=.5,validation_loss=1.,file_sha256={'test_artifact.txt':file_hash(artifact)})
    save_json(c.output/'summary.json',row)
    from exp16a.scheduler import seal
    seal(c);assert verify(c)['status']=='completed'
    artifact.write_text('changed')
    with pytest.raises(ValueError,match='Artifact mismatch'):verify(c)
    import shutil
    shutil.rmtree(c.output)

def test_train_cifar_integrity_does_not_read_test(monkeypatch):
    from exp16a.data import OfflineCIFAR100
    import exp16a.data as data
    checked=[]
    monkeypatch.setattr(data,'check_integrity',lambda path,md5:checked.append(path) or True)
    dataset=OfflineCIFAR100.__new__(OfflineCIFAR100);dataset.train=True;dataset.root='/read-only'
    assert dataset._check_integrity()
    assert len(checked)==1 and checked[0].endswith('/train')
    checked.clear();dataset.train=False;assert dataset._check_integrity()
    assert len(checked)==1 and checked[0].endswith('/test')

def test_three_seed_selection_and_formal_counts():
    from exp16a.search import review_jobs,select
    rows=[dict(**c.values(),trial_id=c.id,status='completed',accuracy=.5+c.lr,validation_loss=1.,official_evaluation_used=False) for c in coarse_jobs()]
    top=top_two(rows);reviews=review_jobs(top);assert len(reviews)==112
    reviewed=[dict(**c.values(),trial_id=c.id,status='completed',accuracy=.7,validation_loss=.8,official_evaluation_used=False) for c in reviews]
    winners,evidence=select(top,rows,reviewed)
    assert sum(len(x) for x in winners.values())==28
    assert sum(10 for group in winners.values() for _ in group)==280
    for task in winners:
        for method,v in winners[task].items():
            assert v['task']==task and v['method']==method
            assert all(not row['official_evaluation_used'] for row in reviewed)

def test_report_student_t_statistics():
    from exp16a.report import stats
    values=np.arange(10.)
    out=stats(values)
    assert out['mean']==4.5 and out['sd']==pytest.approx(np.std(values,ddof=1))
    from scipy.stats import t
    assert out['ci95_high']==pytest.approx(4.5+t.ppf(.975,9)*np.std(values,ddof=1)/np.sqrt(10))
