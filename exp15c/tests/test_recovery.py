"""Exercise recovery, new failures, selection, reporting, and strict provenance."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
from exp15c import RESULTS,ROOT
from exp15c import config,scheduler,search,report
from exp15c.config import Trial,SEARCH_SEED,REVIEW_SEEDS,FINAL_SEEDS,save_json,json_hash,file_hash
from exp15c.support import code_hashes,read


@pytest.fixture
def trials(monkeypatch,tmp_path):
    monkeypatch.setattr(config,'RESULTS',tmp_path)
    monkeypatch.setattr(scheduler,'RESULTS',tmp_path)
    def write(cfg,status='completed'):
        cfg.output.mkdir(parents=True,exist_ok=True)
        resolved=dict(cfg.protocol(),code_sha256=code_hashes())
        digest=json_hash(resolved);resolved['config_sha256']=digest
        save_json(cfg.output/'config.json',resolved)
        (cfg.output/'train.log').write_text('original training log\n')
        row=dict(**cfg.values(),trial_id=cfg.id,status=status)
        if status=='completed':
            (cfg.output/'checkpoint.pt').write_bytes(b'fixture checkpoint')
            row.update(accuracy=.7,validation_loss=.6,completed_epochs=5,official_evaluation_used=cfg.stage=='final',
                code_sha256=code_hashes(),config_sha256=digest,
                file_sha256={name:file_hash(cfg.output/name) for name in ('config.json','checkpoint.pt')})
        else:
            row.update(error='FloatingPointError: Nonfinite gradient' if status=='non_finite' else 'CUDA out of memory' if status=='oom' else 'RuntimeError: model loading failed',
                traceback='fixture traceback')
        save_json(cfg.output/'summary.json',row)
        return row
    return write


def job(lr=.01,stage='search'):
    seed=dict(search=SEARCH_SEED,review=REVIEW_SEEDS[0],final=FINAL_SEEDS[0])[stage]
    return Trial('cv','momentum-no-noise',lr,stage=stage,seed=seed)


def fake_workers(monkeypatch,write,outcomes):
    launched=[]
    def spawn(argv,**kwargs):
        cfg=Trial(**json.loads(argv[-1]));launched.append(cfg.id)
        status=outcomes[cfg.lr];write(cfg,status)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] in ('1','2','3')
        return SimpleNamespace(pid=100+len(launched),poll=lambda:0 if status=='completed' else 1)
    monkeypatch.setattr(scheduler.subprocess,'Popen',spawn)
    monkeypatch.setattr(scheduler,'idle',lambda gpu:True)
    monkeypatch.setattr(scheduler.time,'sleep',lambda _:None)
    return launched


@pytest.mark.parametrize('stage',['search','review'])
def test_existing_and_new_divergence_continue_without_rerunning(monkeypatch,trials,stage):
    done,old_bad,new_bad,new_good=[job(lr,stage) for lr in (.01,.02,.03,.04)]
    trials(done);trials(old_bad,'non_finite')
    before={str(p):file_hash(p) for c in (done,old_bad) for p in c.output.iterdir()}
    launched=fake_workers(monkeypatch,trials,{.03:'non_finite',.04:'completed'})
    rows=scheduler.run_batch(stage,[done,old_bad,new_bad,new_good])
    assert launched==[new_bad.id,new_good.id]
    assert [r['status'] for r in rows]==['completed','non_finite','non_finite','completed']
    assert all(file_hash(Path(p))==h for p,h in before.items())
    state=read(scheduler.RESULTS/f'{stage}_status.json')
    assert state[f'{old_bad.id}:primary']['status']=='non_finite'
    assert state[f'{old_bad.id}:primary']['error']=='FloatingPointError: Nonfinite gradient'
    assert state[f'{new_bad.id}:primary']['traceback']=='fixture traceback'


@pytest.mark.parametrize('stage,outcome',[('search','oom'),('search','failed'),('review','oom'),('final','non_finite'),('final','failed')])
def test_new_engineering_errors_and_formal_divergence_stop(monkeypatch,trials,stage,outcome):
    jobs=[job(lr,stage) for lr in (.01,.02,.03,.04)]
    launched=fake_workers(monkeypatch,trials,{.01:outcome,.02:'completed',.03:'completed',.04:'completed'})
    with pytest.raises(RuntimeError,match='failed'):
        scheduler.run_batch(stage,jobs)
    assert jobs[3].id not in launched
    state=read(scheduler.RESULTS/f'{stage}_status.json')
    assert state[f'{jobs[0].id}:primary']['status']=='failed'
    assert state[f'{jobs[3].id}:primary']['status']=='pending'


@pytest.mark.parametrize('stage,outcome',[('search','oom'),('search','failed'),('final','non_finite')])
def test_existing_fatal_results_are_not_retried(monkeypatch,trials,stage,outcome):
    cfg=job(stage=stage);trials(cfg,outcome)
    monkeypatch.setattr(scheduler.subprocess,'Popen',lambda *a,**k:pytest.fail('Retried failed trial'))
    with pytest.raises(RuntimeError,match='inspection'):scheduler.run_batch(stage,[cfg])


def test_real_23_successes_and_one_divergence_reused_without_launch(monkeypatch,tmp_path):
    frozen=search.load_frozen();jobs=[]
    before=read(RESULTS/'recovery/original_search_status.json')
    for key,row in before.items():
        if row['status'] in ('completed','failed'):jobs.append(Trial(**frozen['candidates'][key.split(':')[0]]))
    monkeypatch.setattr(scheduler,'RESULTS',tmp_path)
    monkeypatch.setattr(scheduler.subprocess,'Popen',lambda *a,**k:pytest.fail('Existing trial launched'))
    rows=scheduler.run_batch('reuse_audit',jobs)
    assert sum(r['status']=='completed' for r in rows)==23
    assert sum(r['status']=='non_finite' for r in rows)==1


def test_core_config_and_artifact_tampering_remain_fatal(trials):
    cfg=job();trials(cfg)
    (cfg.output/'checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):scheduler.completed(cfg)
    trials(cfg)
    resolved=read(cfg.output/'config.json');resolved['lr']=2.
    save_json(cfg.output/'config.json',resolved)
    with pytest.raises(ValueError,match='configuration hash'):scheduler.completed(cfg)
    trials(cfg,'non_finite')
    resolved=read(cfg.output/'config.json');resolved.pop('config_sha256')
    resolved['code_sha256']['exp15c/train.py']='changed'
    resolved['config_sha256']=json_hash(resolved);save_json(cfg.output/'config.json',resolved)
    with pytest.raises(ValueError,match='core changed'):scheduler.non_finite(cfg)


def test_rank_ignores_exclusions_and_invalid_metrics():
    base=dict(status='completed',accuracy=.7,validation_loss=.6,trial_id='good')
    rows=[base,dict(base,status='non_finite',trial_id='excluded'),dict(base,accuracy=float('nan'),trial_id='nan'),
        dict(base,validation_loss=float('inf'),trial_id='inf'),dict(base,accuracy=2.,trial_id='invalid')]
    assert search.rank(rows)==[base]


def test_report_accepts_explicit_exclusions_but_rejects_missing_and_errors(trials):
    good,bad=job(.01),job(.02)
    rows=[trials(good),trials(bad,'non_finite')]
    counts,excluded=report.audit_trials('search',[good.values(),bad.values()],rows)
    assert counts==dict(planned=2,completed=1,non_finite=1)
    assert excluded[0]['reason']=='FloatingPointError: Nonfinite gradient'
    with pytest.raises(ValueError,match='Missing planned'):report.audit_trials('search',[good.values(),bad.values()],rows[:1])
    with pytest.raises(RuntimeError):report.audit_trials('search',[bad.values()],[trials(bad,'oom')])
    formal=job(stage='final')
    with pytest.raises(ValueError,match='Invalid exclusion'):report.audit_trials('formal',[formal.values()],[trials(formal,'non_finite')])


def test_phase1_accepts_only_core_unchanged_and_current_tests(monkeypatch,tmp_path):
    monkeypatch.setattr(search,'RESULTS',tmp_path)
    current=code_hashes();old=dict(current);old['exp15c/scheduler.py']='old';old['exp15c/report.py']='old'
    save_json(tmp_path/'phase1_acceptance.json',dict(status='passed',code_sha256=old))
    (tmp_path/'tests.log').write_text('passed\n')
    tests=dict(status='passed',code_sha256=current,log_sha256=file_hash(tmp_path/'tests.log'))
    save_json(tmp_path/'test_results.json',tests)
    search.require_phase1()
    old['exp15c/optimizer.py']='changed'
    save_json(tmp_path/'phase1_acceptance.json',dict(status='passed',code_sha256=old))
    with pytest.raises(ValueError,match='core'):search.require_phase1()
    old['exp15c/optimizer.py']=current['exp15c/optimizer.py']
    save_json(tmp_path/'phase1_acceptance.json',dict(status='passed',code_sha256=old))
    (tmp_path/'tests.log').write_text('tampered')
    with pytest.raises(ValueError,match='log changed'):search.require_phase1()


def test_frozen_grid_hash_still_enforced(monkeypatch,tmp_path):
    frozen=search.load_frozen()
    path=tmp_path/'grid.json';save_json(path,frozen)
    (tmp_path/'extended_grid.sha256').write_text(file_hash(path))
    monkeypatch.setattr(search,'RESULTS',tmp_path);monkeypatch.setattr(search,'FROZEN',path)
    assert search.load_frozen()==frozen
    save_json(path,dict(frozen,counts={}))
    with pytest.raises(ValueError,match='Frozen extension changed'):search.load_frozen()


@pytest.mark.parametrize('all_invalid',[False,True])
def test_review_divergence_excluded_and_no_valid_review_stops(monkeypatch,tmp_path,all_invalid):
    jobs=[job(.01),job(.02)];ids=[c.id for c in jobs]
    frozen=dict(candidates={c.id:c.values() for c in jobs},schemes={s:{'cv:momentum-no-noise':ids} for s in ('A','B')})
    path=tmp_path/'grid.json';save_json(path,frozen)
    for name,value in [('RESULTS',tmp_path),('FROZEN',path),('REVIEW',tmp_path/'review.json'),('SELECTED',tmp_path/'selected.json')]:
        monkeypatch.setattr(search,name,value)
    monkeypatch.setattr(search,'load_frozen',lambda:frozen)
    from exp15c import pipeline
    monkeypatch.setattr(pipeline,'status',lambda *a,**k:None)
    def run(name,trials):
        rows=[]
        for c in trials:
            bad=name=='review' and (c.lr==.01 or all_invalid)
            rows.append(dict(**c.values(),trial_id=c.id,status='non_finite' if bad else 'completed',
                error='FloatingPointError: Nonfinite gradient' if bad else None,
                completed_epochs=0 if bad else 5,official_evaluation_used=False,
                accuracy=None if bad else .9 if c.lr==.01 else .8,validation_loss=None if bad else .5))
        return rows
    monkeypatch.setattr(search,'run_batch',run)
    if all_invalid:
        with pytest.raises(ValueError,match='No three-seed stable candidate'):search.search_and_review()
    else:
        final=search.search_and_review()
        assert len(final)==10 and all(c.lr==.02 for c in final)
        rankings=read(tmp_path/'review_results.json')['rankings']
        assert all(len(r['excluded'])==1 for r in rankings.values())


def test_no_valid_search_candidate_stops(monkeypatch,tmp_path):
    cfg=job();frozen=dict(candidates={cfg.id:cfg.values()},schemes={'A':{'cv:momentum-no-noise':[cfg.id]}})
    monkeypatch.setattr(search,'RESULTS',tmp_path);monkeypatch.setattr(search,'load_frozen',lambda:frozen)
    monkeypatch.setattr(search,'run_batch',lambda *a:[dict(**cfg.values(),trial_id=cfg.id,status='non_finite',error='FloatingPointError: Nonfinite gradient')])
    with pytest.raises(ValueError,match='No stable candidate: A cv:momentum-no-noise'):search.search_and_review()
