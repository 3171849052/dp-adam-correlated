from dataclasses import replace
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from exp10b import BASE,ROOT
from exp10b.config import Trial,LAMBDAS,COUNTS,FINAL_SEEDS,save_json,file_hash
from exp10b.protocol import plan,search_jobs,choose_winners,formal_jobs,validate_formal
from exp10b.report import statistics,paired_differences,summarize

def internal_rows():
    return [dict(j.values(),trial_id=j.id,status='completed',accuracy=.7-j.module/100,
        validation_loss=.5+j.module/100,completed_epochs=5,optimizer_steps=j.total_steps,
        official_evaluation_used=False) for s in ('stage1','stage2') for j in search_jobs(s)]

def test_exact_budget_seed_pairing_and_positive_selection():
    assert len(search_jobs('stage1'))==36 and len(search_jobs('stage2'))==84
    rows=internal_rows();winners,evidence=choose_winners(rows)
    assert all(winners[t]['module']==.01 for t in winners)
    assert all(evidence[t]['global_best']['module']==0 for t in evidence)
    assert all(len(g['seeds'])==10 for e in evidence.values() for g in e['all_lambdas'])
    final=formal_jobs(winners)
    for cfg in final: validate_formal(cfg,winners)
    with pytest.raises(AssertionError): validate_formal(Trial('cv',.3,20261111,'final'),winners)
    assert len(final)==len({j.id for j in final})==40 and len(rows)+len(final)==160
    for task in ('cv','nlp'):
        assert {(j.module,j.seed) for j in final if j.task==task}=={(lam,s) for lam in (0.,.01) for s in FINAL_SEEDS}
    for task in ('cv','nlp'):
        configs=[j.protocol() for j in search_jobs('stage1') if j.task==task and j.seed==20261101]
        assert len({json.dumps(c['optimizer'],sort_keys=True) for c in configs})==1
        assert len({json.dumps(c['privacy'],sort_keys=True) for c in configs})==1
    with pytest.raises(AssertionError): choose_winners(rows[:-1])
    bad=copy.deepcopy(rows);bad[0]['official_evaluation_used']=True
    with pytest.raises(AssertionError): choose_winners(bad)
    ties=copy.deepcopy(rows)
    for r in ties: r['accuracy']=.7
    assert all(w['module']==.01 for w in choose_winners(ties)[0].values())

def test_CI_and_formal_positive_minus_zero():
    stat=statistics(range(10))
    assert (stat['ci95_high']-stat['mean'])/stat['se']==pytest.approx(2.2621571628)
    winners,_=choose_winners(internal_rows())
    rows=[dict(j.values(),accuracy=.7+(.01 if j.module>0 else 0),validation_loss=.5,
        seconds=1.,peak_allocated_bytes=100.,peak_reserved_bytes=200.) for j in formal_jobs(winners)]
    raw,paired=paired_differences(rows,winners)
    assert len(raw)==100 and len(paired)==10
    assert all(r['mean']==pytest.approx(.01) for r in paired if r['metric']=='accuracy')
    assert all(r['n']==10 for r in summarize(rows))

def test_plan_CLI_never_trains_and_rejects_GPU_zero():
    before=sorted(str(p) for p in (BASE/'results').rglob('summary.json'))
    result=subprocess.run([sys.executable,'-B','-m','exp10b.run','--gpus','1','2','3','--plan'],
        cwd=ROOT,capture_output=True,text=True,check=True)
    output=json.loads(result.stdout)
    assert output['counts']==COUNTS and len(output['stage1'])==36 and len(output['stage2'])==84
    assert output['stage4']['trials']==40 and output['excluded_gpus']==[0]
    assert len(output['stage1'])+len(output['stage2'])+len(output['stage4']['planned_trials'])==160
    assert before==sorted(str(p) for p in (BASE/'results').rglob('summary.json'))
    with pytest.raises(AssertionError): plan([0,1,2,3])
    with pytest.raises(AssertionError): plan([1,1])
    result=subprocess.run([sys.executable,'-B','-m','exp10b.run','--gpus','0','--plan'],cwd=ROOT,capture_output=True)
    assert result.returncode!=0

def test_fifo_exclusive_capacity_reuse_and_failure(tmp_path,monkeypatch):
    from exp10b import config,launcher
    from exp10b.audit import audit_fifo
    monkeypatch.setattr(config,'RESULTS',tmp_path);monkeypatch.setattr(launcher,'RESULTS',tmp_path)
    worker=tmp_path/'worker.py';worker.write_text('''import json,os,sys,time
from pathlib import Path
j=json.loads(sys.argv[1]);d=Path(sys.argv[2]);time.sleep(.3)
j.update(status='completed',trial_id=sys.argv[3],gpu=os.environ['CUDA_VISIBLE_DEVICES'])
(d/'summary.json').write_text(json.dumps(j))
''')
    jobs=[Trial('cv',lam,stage='smoke') for lam in LAMBDAS[:4]]
    jobs += [Trial('nlp',lam,stage='smoke') for lam in LAMBDAS]
    jobs += [Trial('cv',1.,stage='smoke')]
    verify=lambda cfg,**kw:json.loads((cfg.output/'summary.json').read_text())
    command=lambda j:[sys.executable,'-B',str(worker),json.dumps(j.values()),str(j.output),j.id]
    rows=launcher.run_queue(jobs,command=command,verify=verify,poll_seconds=.01)
    assert {r['gpu'] for r in rows}=={'1','2','3'}
    events=[json.loads(s) for s in (tmp_path/'scheduler.jsonl').read_text().splitlines()]
    assert [e['trial_id'] for e in events if e['event']=='start']==[j.id for j in jobs]
    assert audit_fifo(tmp_path/'scheduler.jsonl')['status']=='passed'
    parallel=[replace(j,seed=20261102) for j in jobs if j.task=='nlp']
    launcher.run_queue(parallel,command=command,verify=verify,poll_seconds=.01)
    assert all(audit_fifo(tmp_path/'scheduler.jsonl')['peak_concurrency'][g]['nlp']==2 for g in (1,2,3))
    with monkeypatch.context() as patch:
        patch.setattr(launcher.subprocess,'Popen',lambda *a,**k:pytest.fail('Completed jobs must be reused'))
        assert launcher.run_queue(jobs,command=command,verify=verify)==rows
    fail=[Trial('cv',.3,20261110,'smoke'),Trial('nlp',.3,20261110,'smoke')]
    with pytest.raises(RuntimeError,match='Worker failure'):
        launcher.run_queue(fail,gpus=[1],command=lambda j:[sys.executable,'-c','raise RuntimeError("test failure")'],
                          verify=verify,poll_seconds=.01)
    assert not fail[1].output.exists()

def test_freeze_code_matrix_evidence_and_immutability(tmp_path,monkeypatch):
    from exp10b import frozen
    monkeypatch.setattr(frozen,'RESULTS',tmp_path)
    monkeypatch.setattr(frozen,'get_matrix',lambda *a:None)
    winners,_=choose_winners(internal_rows())
    for name in ('assets_manifest.json','full_plan.json','stage1_results.json','stage2_results.json',
                 'selection_evidence.json','platform_validation.json'):
        save_json(tmp_path/name,{'test':True})
    for row in internal_rows(): save_json(tmp_path/row['stage']/row['task']/row['trial_id']/'summary.json',row)
    (tmp_path/'matrix_cache').mkdir();(tmp_path/'matrix_cache/example.json').write_text('{"test":true}')
    assert frozen.freeze(winners)==winners
    changed=copy.deepcopy(winners);changed['cv']['module']=.3
    with pytest.raises(AssertionError,match='immutable'): frozen.freeze(changed)
    with monkeypatch.context() as patch:
        patch.setattr(frozen,'code_hashes',lambda:{'changed':'source'})
        with pytest.raises(AssertionError,match='source changed'): frozen.load_frozen()
    (tmp_path/'matrix_cache/example.json').write_text('{"test":false}')
    with pytest.raises(AssertionError,match='evidence changed'): frozen.load_frozen()

def test_historical_sources_and_runtime_paths_read_only():
    from exp10b.audit import protected_snapshot
    snapshot=protected_snapshot()
    assert not any(p.startswith('exp10b/') for p in snapshot)
    provenance=json.loads((BASE/'kernel_provenance.json').read_text())
    for entry in provenance.values(): assert file_hash(ROOT/entry['source'])==entry['sha256']
    for name in ('cv_model.py','nlp_model.py','privacy.py','bandinvmf.py','clipping.py'):
        assert (BASE/name).read_text()==(ROOT/'exp9'/name).read_text().replace('exp9','exp10b')
    for key in ('TMPDIR','HF_HOME','HF_DATASETS_CACHE','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR'):
        assert Path(os.environ[key]).is_relative_to(BASE)
    assert (Trial('cv',0).total_steps,Trial('cv',0,stage='smoke',smoke_phase='final').total_steps,Trial('nlp',0).total_steps)==(225,250,310)
