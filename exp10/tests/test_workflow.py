from dataclasses import replace
import json
import os
import subprocess
import sys
import pytest
from exp10 import BASE,ROOT
from exp10.config import Trial,METHODS,COUNTS
from exp10.search import initial_jobs,refine,top_two,recheck_jobs,choose_winners,formal_jobs,rank
from exp10.stage2 import plan
from exp10.report import statistics,summarize

def synthetic_search():
    rows=[]
    for i,j in enumerate(initial_jobs()): rows.append(dict(j.values(),trial_id=j.id,status='completed',accuracy=.6+i/10000,validation_loss=1-i/10000))
    for task in ('cv','nlp'):
        for method in METHODS:
            jobs,_=refine([r for r in rows if r['task']==task and r['method']==method][:24])
            rows += [dict(j.values(),trial_id=j.id,status='completed',accuracy=.75,validation_loss=.5) for j in jobs]
    return rows

def test_budget_distinct_refinements_review_and_formal_pairing():
    rows=synthetic_search();assert len(rows)==len({r['trial_id'] for r in rows})==120
    top2=top_two(rows);jobs=recheck_jobs(top2);assert len(jobs)==24
    rechecks=[dict(j.values(),trial_id=j.id,status='completed',accuracy=.7,validation_loss=.6) for j in jobs]
    winners,evidence=choose_winners(top2,rows,rechecks);final=formal_jobs(winners)
    assert len(final)==len({j.id for j in final})==40 and len(rows)+len(jobs)+len(final)==184
    for task in ('cv','nlp'):
        for method in METHODS:
            candidates=[r for r in rows if r['task']==task and r['method']==method]
            assert len(candidates)==len({(r['lr'],r['C'],r['module']) for r in candidates})==30
            assert all(len(c['trial_ids'])==4 for c in evidence[task][method])
    formal=[dict(j.values(),accuracy=.7+(.01 if j.method==METHODS[0] else 0),validation_loss=.5,actual_epsilon=8.,seconds=1.,peak_allocated_bytes=100.) for j in final]
    stats,paired=summarize(formal);assert all(s['n']==10 for s in stats+paired)
    assert all(p['mean']==pytest.approx(.01) for p in paired if p['metric']=='accuracy')

def test_accuracy_loss_id_ties_and_ci():
    rows=[dict(status='completed',accuracy=.7,validation_loss=loss,trial_id=id) for id,loss in [('z',.4),('b',.3),('a',.3)]]
    assert [r['trial_id'] for r in rank(rows)]==['a','b','z']
    stat=statistics(range(10));assert (stat['ci95_high']-stat['mean'])/stat['se']==pytest.approx(2.2621571628)
def test_plan_has_exact_initial_pairs_and_never_trains():
    args=[sys.executable,'-B','-m','exp10.stage2','--gpus','1','2','3','--plan']
    before=sorted(str(p) for p in (BASE/'results').rglob('summary.json'))
    result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,check=True);p=json.loads(result.stdout)
    assert p['counts']==COUNTS and p['excluded_gpus']==[0] and len(p['initial_trials'])==96
    assert sorted(str(p) for p in (BASE/'results').rglob('summary.json'))==before
    for task in ('cv','nlp'):
        a={(j.lr,j.C) for j in initial_jobs() if j.task==task and j.method==METHODS[0]}
        b={(j.lr,j.C) for j in initial_jobs() if j.task==task and j.method==METHODS[1]};assert a==b
    with pytest.raises(AssertionError): plan([0,1,2,3])
    reject=subprocess.run([sys.executable,'-B','-m','exp10.stage2','--gpus','0','--plan'],cwd=ROOT,capture_output=True)
    assert reject.returncode!=0

def test_real_fifo_capacity_gpu_exclusion_and_completed_reuse(tmp_path,monkeypatch):
    from exp10 import config,launcher
    from exp10.audit import audit_fifo
    monkeypatch.setattr(config,'RESULTS',tmp_path);monkeypatch.setattr(launcher,'RESULTS',tmp_path)
    worker=tmp_path/'worker.py';worker.write_text('''import json,os,sys,time
from pathlib import Path
j=json.loads(sys.argv[1]);d=Path(sys.argv[2]);time.sleep(.3)
j.update(status='completed',trial_id=sys.argv[3],gpu=os.environ['CUDA_VISIBLE_DEVICES'])
(d/'summary.json').write_text(json.dumps(j))
''')
    jobs=[Trial('cv',METHODS[0],.001+i*.00001,30,.3,stage='smoke') for i in range(4)]
    jobs += [Trial('nlp',METHODS[1],.001+i*.00001,20,2,stage='smoke') for i in range(6)]
    jobs += [Trial('cv',METHODS[1],.002,30,2,stage='smoke')]
    verify=lambda cfg,**kw:json.loads((cfg.output/'summary.json').read_text())
    command=lambda j:[sys.executable,'-B',str(worker),json.dumps(j.values()),str(j.output),j.id]
    rows=launcher.run_queue(jobs,command=command,verify=verify,poll_seconds=.01)
    assert {r['gpu'] for r in rows}=={'1','2','3'}
    events=[json.loads(s) for s in (tmp_path/'scheduler.jsonl').read_text().splitlines()]
    assert [e['trial_id'] for e in events if e['event']=='start']==[j.id for j in jobs]
    assert audit_fifo(tmp_path/'scheduler.jsonl')['status']=='passed'
    parallel=[replace(j,seed=20261102) for j in jobs if j.task=='nlp']
    launcher.run_queue(parallel,command=command,verify=verify,poll_seconds=.01)
    fifo=audit_fifo(tmp_path/'scheduler.jsonl');assert all(fifo['peak_concurrency'][g]['nlp']==2 for g in (1,2,3))
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**k:pytest.fail('Completed jobs must be reused'))
    assert launcher.run_queue(jobs,command=command,verify=verify)==rows

def test_freeze_source_matrix_evidence_and_immutability(tmp_path,monkeypatch):
    from exp10 import frozen
    from exp10.config import save_json
    from exp10.matrices import get_matrix
    monkeypatch.setattr(frozen,'RESULTS',tmp_path)
    rows=synthetic_search();top2=top_two(rows)
    rechecks=[dict(j.values(),trial_id=j.id,status='completed',accuracy=.7,validation_loss=.6) for j in recheck_jobs(top2)]
    winners,_=choose_winners(top2,rows,rechecks)
    for name in ('assets_manifest.json','initial_search_configs.json','initial_search_results.json','refinement_configs.json',
        'refinement_decisions.json','search_results.json','recheck_results.json','top2.json','selection_evidence.json','platform_validation.json'):
        save_json(tmp_path/name,{'test':True})
    for row in rows+rechecks: save_json(tmp_path/row['stage']/row['task']/row['trial_id']/'summary.json',row)
    (tmp_path/'matrix_cache').mkdir();(tmp_path/'matrix_cache/example.json').write_text('{"test":true}')
    assert frozen.freeze(winners)==winners
    changed=__import__('copy').deepcopy(winners);changed['cv'][METHODS[0]]['lr']=.1
    with pytest.raises(AssertionError,match='immutable'): frozen.freeze(changed)
    (tmp_path/'matrix_cache/example.json').write_text('{"test":false}')
    with pytest.raises(AssertionError,match='evidence changed'): frozen.load_frozen()

def test_protected_roots_and_kernel_provenance():
    from exp10.audit import protected_snapshot
    from exp10.config import file_hash
    snapshot=protected_snapshot()
    assert not any(p.startswith('exp10/') for p in snapshot)
    manifest=json.loads((BASE/'kernel_provenance.json').read_text())
    for name,entry in manifest.items(): assert file_hash(ROOT/entry['source'])==entry['sha256']
    for name in ('cv_model.py','nlp_model.py','privacy.py','noise.py','bandinvmf.py','clipping.py'):
        assert (BASE/name).read_text()==(ROOT/'exp9'/name).read_text().replace('exp9','exp10')
    for name in ('TMPDIR','HF_HOME','HF_DATASETS_CACHE','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR'):
        assert __import__('pathlib').Path(os.environ[name]).is_relative_to(BASE)
