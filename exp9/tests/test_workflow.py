from dataclasses import replace
import json
import os
import subprocess
import sys
import numpy as np
import pytest
from exp9 import BASE, ROOT
from exp9.config import Trial, METHODS, IID, MF, FINAL_SEEDS, COUNTS, save_json
from exp9.grid import grid,all_grid
from exp9.search import top_two,recheck_jobs,choose_winners
from exp9.final import formal_jobs
from exp9.report import statistics,summarize

def test_exact_grid_and_output_protocol():
    assert len(all_grid())==321
    for task,size in (('cv',177),('nlp',144)):
        jobs=grid(task);assert len(jobs)==size
        assert len({j.id for j in jobs})==size
        assert len([j for j in jobs if j.method==IID])==9
        assert all(j.seed==20261101 and j.epsilon==8 for j in jobs)
        spaces=[{(j.lr,j.C,j.eps_scale) for j in jobs if j.method==m} for m in MF]
        assert spaces[0]==spaces[1]==spaces[2]
        spaces=[{(j.lr,j.C,j.eps_scale) for j in jobs if j.method==m+'-scale'} for m in MF]
        assert spaces[0]==spaces[1]==spaces[2]
    for key in ('TMPDIR','HF_HOME','HF_DATASETS_CACHE','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR'):
        from pathlib import Path
        assert Path(os.environ[key]).is_relative_to(BASE)
    cfg=grid('cv')[0]
    assert cfg.total_steps==225 and replace(cfg,stage='final',seed=FINAL_SEEDS[0]).total_steps==250
    assert all(j.total_steps==310 and j.physical_batch==1000 for j in grid('nlp'))

def synthetic_search():
    return [dict(j.values(),trial_id=j.id,status='completed',accuracy=.6+i/10000) for i,j in enumerate(all_grid())]

def test_independent_top2_review_and_formal_counts():
    rows=synthetic_search();top2=top_two(rows);jobs=recheck_jobs(top2)
    assert len(jobs)==56 and len({j.id for j in jobs})==56
    rechecks=[dict(j.values(),trial_id=j.id,status='completed',accuracy=.7) for j in jobs]
    winners,evidence=choose_winners(top2,rows,rechecks)
    assert sum(len(winners[t]) for t in winners)==14
    for task in winners:
        for method in METHODS:
            assert len(evidence[task][method])==2
            assert all(len(c['trial_ids'])==3 for c in evidence[task][method])
    finals=formal_jobs(winners);sweep=formal_jobs(winners,'sweep')
    assert len(finals)==140 and len(sweep)==126
    assert len(rows)+len(jobs)+len(finals)+len(sweep)==643
    assert len({j.id for j in finals+sweep})==266
    for j in finals+sweep:
        assert all(getattr(j,k)==winners[j.task][j.method][k] for k in ('lr','C','eps_scale'))
    rechecks[0]['status']='non_finite'
    with pytest.raises(AssertionError): choose_winners(top2,rows,rechecks)

def test_final_epoch_accuracy_ranking_and_statistics():
    rows=synthetic_search();first=next(r for r in rows if r['method']==IID and r['task']=='cv')
    first.update(accuracy=.99,loss=999,early_epoch_accuracy=0.)
    top2=top_two(rows)
    assert Trial(**top2['cv'][IID][0]).id==first['trial_id']
    values=np.linspace(.7,.79,10);st=statistics(values)
    assert st['mean']==pytest.approx(.745) and st['std']==pytest.approx(values.std(ddof=1))
    assert (st['ci95_high']-st['mean'])/st['se']==pytest.approx(2.2621571628)
    winners,_=choose_winners(top2,rows,[dict(j.values(),trial_id=j.id,status='completed',accuracy=.7) for j in recheck_jobs(top2)])
    formal=[dict(j.values(),trial_id=j.id,status='completed',accuracy=.7+(.01 if j.method!=IID else 0)) for j in formal_jobs(winners)]
    summary,paired=summarize(formal)
    assert len(summary)==14 and len(paired)==42 and all(s['n']==10 for s in summary)
    assert any(p['mean']==pytest.approx(.01) and p['wins']==10 for p in paired)

def test_cli_plan_has_exact_protocol_without_launching():
    path=BASE/'results/stage2_plan.json'
    before=path.read_bytes() if path.exists() else None
    result=subprocess.run([sys.executable,'-B','-m','exp9.stage2','--gpus','0','1','2','3',
                           '--cv-per-gpu','1','--nlp-per-gpu','2','--plan'],cwd=ROOT,capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout);assert plan['counts']==COUNTS and plan['gpus']==[0,1,2,3]
    assert plan['stage_order']==['search','recheck','freeze','final','sweep','report']
    assert (path.read_bytes() if path.exists() else None)==before

def test_real_subprocess_fifo_capacity_mapping_and_reuse(tmp_path,monkeypatch):
    from exp9 import config,launcher
    from exp9.audit import audit_fifo
    monkeypatch.setattr(config,'RESULTS',tmp_path);monkeypatch.setattr(launcher,'RESULTS',tmp_path)
    worker=tmp_path/'worker.py'
    worker.write_text('''import json,os,sys,time
from pathlib import Path
j=json.loads(sys.argv[1]);d=Path(sys.argv[2]);time.sleep(.25)
j.update(status='completed',trial_id=sys.argv[3],gpu=os.environ['CUDA_VISIBLE_DEVICES'])
(d/'summary.json').write_text(json.dumps(j))
''')
    jobs=[Trial('cv',IID,.001+i*.00001,10,stage='smoke') for i in range(5)]
    jobs += [Trial('nlp',IID,.001+i*.00001,1,stage='smoke') for i in range(8)]
    jobs += [Trial('cv',IID,.002+i*.00001,10,stage='smoke') for i in range(2)]
    verify=lambda cfg,**kw:json.loads((cfg.output/'summary.json').read_text())
    command=lambda j:[sys.executable,'-B',str(worker),json.dumps(j.values()),str(j.output),j.id]
    rows=launcher.run_queue(jobs,command=command,verify=verify,poll_seconds=.01)
    events=[json.loads(s) for s in (tmp_path/'scheduler.jsonl').read_text().splitlines()]
    assert [e['trial_id'] for e in events if e['event']=='start']==[j.id for j in jobs]
    assert {r['gpu'] for r in rows}=={'0','1','2','3'}
    assert audit_fifo(tmp_path/'scheduler.jsonl')['status']=='passed'
    active={};peak={g:0 for g in range(4)}
    for e in events:
        if e['event']=='start':
            active[e['trial_id']]=e
            if e['task']=='nlp': peak[e['gpu']]=max(peak[e['gpu']],sum(a['gpu']==e['gpu'] for a in active.values()))
        elif e['event']=='finish': active.pop(e['trial_id'])
    assert max(peak.values())==2 and all(v<=2 for v in peak.values())
    # A CV-to-NLP transition may leave only some GPUs available initially.
    # A standalone NLP wave must fill both slots on every GPU.
    parallel=[replace(j,seed=20261102) for j in jobs if j.task=='nlp']
    launcher.run_queue(parallel,command=command,verify=verify,poll_seconds=.01)
    full=audit_fifo(tmp_path/'scheduler.jsonl')
    assert all(full['peak_concurrency'][gpu]['nlp']==2 for gpu in range(4))
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**k:pytest.fail('Completed results must be reused'))
    assert launcher.run_queue(jobs,command=command,verify=verify)==rows

def test_freeze_integrity_and_immutability(tmp_path,monkeypatch):
    from exp9 import frozen
    monkeypatch.setattr(frozen,'RESULTS',tmp_path)
    rows=synthetic_search();top2=top_two(rows)
    rechecks=[dict(j.values(),trial_id=j.id,status='completed',accuracy=.7) for j in recheck_jobs(top2)]
    winners,_=choose_winners(top2,rows,rechecks)
    for name in ('assets_manifest.json','cv_grid.json','nlp_grid.json','grid_results.json',
                 'recheck_results.json','top2.json','selection_evidence.json','platform_validation.json'):
        save_json(tmp_path/name,{'test':True})
    for row in rows+rechecks:
        save_json(tmp_path/row['stage']/row['task']/row['trial_id']/'summary.json',row)
    assert frozen.freeze(winners)==winners
    changed=__import__('copy').deepcopy(winners);changed['cv'][IID]['lr']=.1
    with pytest.raises(AssertionError,match='immutable'): frozen.freeze(changed)
    (tmp_path/'frozen_configs.json').write_text((tmp_path/'frozen_configs.json').read_text()+' ')
    with pytest.raises(AssertionError): frozen.load_frozen()

@pytest.mark.parametrize('fail',[None,'gate','search','freeze','final','sweep'])
def test_stage2_cli_dependency_gates_without_training(fail,tmp_path,monkeypatch):
    from exp9 import stage2, audit, stage1, grid as grids, search, frozen, data, final, report
    monkeypatch.setattr(stage2,'RESULTS',tmp_path)
    calls=[]
    def wrapped(name,result=None):
        def invoke(*args,**kwargs):
            calls.append(name)
            if name==fail: raise RuntimeError('injected dependency failure')
            if name in ('search','final','sweep'):
                assert kwargs==dict(gpus=[0,1,2,3],cv_per_gpu=1,nlp_per_gpu=2)
            return result
        return invoke
    monkeypatch.setattr(audit,'verify_stage1',wrapped('gate'))
    monkeypatch.setattr(stage1,'unit_tests',wrapped('tests'))
    monkeypatch.setattr(grids,'write_grid',wrapped('grid'))
    monkeypatch.setattr(search,'search',wrapped('search',{}))
    monkeypatch.setattr(frozen,'freeze',wrapped('freeze'))
    monkeypatch.setattr(data,'prepare_official',wrapped('official'))
    monkeypatch.setattr(final,'run_formal',lambda stage,**kw:wrapped(stage)(**kw))
    monkeypatch.setattr(report,'generate_report',wrapped('report'))
    monkeypatch.setattr(sys,'argv',['exp9.stage2','--gpus','0','1','2','3','--cv-per-gpu','1','--nlp-per-gpu','2'])
    if fail:
        with pytest.raises(RuntimeError,match='injected'): stage2.main()
        assert calls[-1]==fail
    else:
        stage2.main()
        assert calls==['gate','tests','grid','search','freeze','official','final','sweep','report']
