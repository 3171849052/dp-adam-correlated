from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from exp15d import ROOT
from exp15d.config import Trial, METHODS, EPS_GRID, SEARCH_SEED, inherited_eps
from exp15d import search
from exp15d.audit import equivalent_sources
from exp15d.optimizer import Optimizer
from exp15d.privacy import calibrate
from exp15d.support import matrices


@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('eps',EPS_GRID)
def test_wide_eps_exact_historical_optimizer_and_first_denominator(method,eps):
    namespace={}
    source=(ROOT/'exp15b/optimizer.py').read_text().replace('from exp15b.config import METHODS',f'METHODS={METHODS!r}')
    exec(compile(source,'exp15b_readonly_optimizer','exec'),namespace)
    p=torch.nn.Parameter(torch.tensor([1.,-2.],dtype=torch.float64));q=torch.nn.Parameter(p.detach().clone())
    a=Optimizer([p],method,.5,eps_scale=eps);b=namespace['Optimizer']([q],method,.5,eps_scale=eps)
    for t in range(4):
        a.freeze_scale();b.freeze_scale()
        if t==0:torch.testing.assert_close(p._logical_denominator,torch.full_like(p,eps),rtol=0,atol=0)
        p.grad=torch.tensor([.3*t,.2],dtype=p.dtype);q.grad=p.grad.clone();a.step();b.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        for key,v in a.state[p].items():
            if isinstance(v,torch.Tensor):torch.testing.assert_close(v,b.state[q][key],rtol=0,atol=0)
            else:assert v==b.state[q][key]


def test_mf_eps_invariance_R_recalibration_and_source_equivalence():
    equivalent_sources()
    for task in ('cv','nlp'):
        c=Trial(task,'stpex',.5,eps_scale=1e-8)
        a=matrices(c);b=matrices(replace(c,eps_scale=1.))
        for x,y in zip(a[:4],b[:4]):np.testing.assert_array_equal(x,y)
        ca=calibrate(a[1],c.protocol());cb=calibrate(b[1],replace(c,eps_scale=1.).protocol())
        assert ca==cb
        assert calibrate(a[1],replace(c,clip=3.).protocol())['innovation_std_sum']==pytest.approx(3*ca['innovation_std_sum'])


def fake_row(c,accuracy):
    return dict(**c.values(),trial_id=c.id,status='completed',accuracy=accuracy,validation_loss=1-accuracy,completed_epochs=5,official_evaluation_used=False)


def test_grid_retention_counts_and_nonfinite_exclusion():
    first=search.stage1_trials();assert len(first)==78 and len({c.id for c in first})==78
    rows=[fake_row(c,i/100) for i,c in enumerate(first)]
    # Non-finite never enters the two selected alternatives.
    rows[-1]=dict(rows[-1],status='non_finite')
    kept=search.retain_eps(rows)
    for key,eps in kept.items():
        assert len(eps)==3 and eps[0]==inherited_eps(key.split(':')[0])
    joint=search.joint_trials(kept)
    assert len(joint)==135
    assert len({c.id for c in first+joint})==195
    assert sum(c.clip==1. for c in joint)==54
    assert set(EPS_GRID)=={c.eps_scale for c in first if c.task=='nlp' and c.method=='stpex'}
    assert Trial('nlp','stpex',.15,eps_scale=.01).eps_scale==.01
    with pytest.raises(ValueError):Trial('cv','stpex',.5,eps_scale=0.)


def test_review_three_seed_reversal_and_cross_scheme_dedup(monkeypatch,tmp_path):
    first=search.stage1_trials();screen=[fake_row(c,.5) for c in first]
    joint=search.joint_trials(search.retain_eps(screen))
    all_trials=list({c.id:c for c in first+joint}.values());rows=[];preferred={}
    for task in ('cv','nlp'):
        for method in METHODS:
            preferred[(task,method)]=[c.id for c in joint if c.task==task and c.method==method and c.clip==1.][:2]
    for c in all_trials:
        ids=preferred[(c.task,c.method)];score=.8 if c.id==ids[0] else .7 if c.id==ids[1] else .1
        rows.append(fake_row(c,score))
    frozen=tmp_path/'search_plan.json';frozen.write_text('{}')
    results=tmp_path/'search_results.json';results.write_text('[]')
    monkeypatch.setattr(search,'RESULTS',tmp_path);monkeypatch.setattr(search,'FROZEN',frozen)
    monkeypatch.setattr(search,'REVIEW',tmp_path/'review_plan.json');monkeypatch.setattr(search,'SELECTED',tmp_path/'selected_configs.json')
    calls=[]
    def batch(name,trials,allow_nonfinite=False):
        calls.append((name,len(trials)));output=[]
        for c in trials:
            original=replace(c,stage='search',seed=SEARCH_SEED)
            score=.2 if original.id==preferred[(c.task,c.method)][0] else .9
            output.append(fake_row(c,score))
        return output
    monkeypatch.setattr(search,'run_batch',batch)
    finals=search.review_and_select(rows,joint)
    assert calls==[('review',24)] and len(finals)==60
    selected=json.loads(search.SELECTED.read_text())
    for w in selected['winners']:
        assert w['trial_id']==preferred[(w['task'],w['method'])][1]
        assert w['accuracy_mean']==pytest.approx((.7+.9+.9)/3)
    assert not selected['official_evaluation_used_for_selection']


def test_paired_statistics_and_rng_rejection():
    from exp15d.report import stats,assert_paired
    assert stats([1,2,3])['sample_std']==1
    r=dict(task='cv',seed=1,initialization_sha256='i',classifier_initialization_sha256='c',train_order_sha256='t',rng_trace=['r'],noise_rng_start='a',noise_rng_end='b')
    assert_paired(r,r)
    with pytest.raises(ValueError,match='Unpaired'):assert_paired(r,dict(r,rng_trace=['different']))


def test_four_gpus_worker_mapping_and_direct_eps():
    import inspect
    from exp15d import scheduler,train
    assert 'for gpu in (0,1,2,3):' in inspect.getsource(scheduler.run_batch)
    assert "('0','1','2','3')" in inspect.getsource(train.run)
    assert "torch.device('cuda:0')" in inspect.getsource(train.run)
    assert 'eps_scale=cfg.eps_scale)' in inspect.getsource(train.run)


def test_recovery_checks_config_code_and_artifacts(monkeypatch,tmp_path):
    from exp15d import scheduler
    from exp15d.config import file_hash,save_json
    from exp15d.support import code_hashes
    c=search.stage1_trials()[0]
    monkeypatch.setattr(Trial,'output',property(lambda _:tmp_path))
    (tmp_path/'artifact').write_text('stable')
    r=dict(**c.values(),trial_id=c.id,status='completed',code_sha256=code_hashes(),file_sha256={'artifact':file_hash(tmp_path/'artifact')})
    save_json(tmp_path/'summary.json',r)
    assert scheduler.completed(c)
    (tmp_path/'artifact').write_text('tampered')
    with pytest.raises(ValueError,match='artifact changed'):scheduler.completed(c)
    r['eps_scale']=1.;save_json(tmp_path/'summary.json',r)
    with pytest.raises(ValueError,match='Config mismatch'):scheduler.completed(c)


def test_unplanned_worker_rejected(monkeypatch):
    monkeypatch.setattr(search,'load_frozen',lambda:{'stage1':[]})
    with pytest.raises(ValueError,match='outside authorized'):search.authorize(search.stage1_trials()[0])


def test_report_generates_all_deliverables_and_historical_controlled_CI(monkeypatch,tmp_path):
    from exp15d import report
    from exp15d.config import save_json,file_hash,FINAL_SEEDS
    out=tmp_path/'results';out.mkdir();history=tmp_path/'history';old=history/'exp15b/results';old.mkdir(parents=True)
    first=search.stage1_trials();kept={f'{t}:{m}':[inherited_eps(t),.3,1.] for t in ('cv','nlp') for m in METHODS}
    joint=search.joint_trials(kept);pool=list({c.id:c for c in first+joint}.values())
    def row(c,score):
        r=fake_row(c,score)
        r.update(file_sha256={},actual_epsilon=8.,evaluation_scope='official evaluation' if c.stage=='final' else 'internal validation',
            initialization_sha256='i',classifier_initialization_sha256='h',train_order_sha256='o',rng_trace=['r'],noise_rng_start='a',noise_rng_end='b')
        return r
    screen=[row(c,.5) for c in first];search_rows=[row(c,.5) for c in pool]
    wins=[];old_wins=[];finals={};historical={}
    for scheme in ('A','B'):
        for task in ('cv','nlp'):
            for method in METHODS:
                c=Trial(task,method,.5 if task=='cv' else .15,clip=1.,eps_scale=.3,stage='search',seed=SEARCH_SEED)
                w=dict(scheme=scheme,task=task,method=method,lr=c.lr,clip=c.clip,eps_scale=c.eps_scale)
                wins.append(w);old_wins.append({k:v for k,v in w.items() if k!='eps_scale'})
                for seed in FINAL_SEEDS:
                    f=replace(c,stage='final',seed=seed);finals[f.id]=row(f,.7+seed%3*.001)
                    b=replace(f,eps_scale=inherited_eps(task));r=row(b,.6);r.pop('eps_scale');historical[b.id]=r
            old_wins.append(dict(scheme=scheme,task=task,method='momentum',lr=.5 if task=='cv' else .15,clip=1.))
            for seed in FINAL_SEEDS:
                r=dict(next(r for r in historical.values() if r['task']==task and r['seed']==seed),method='momentum',trial_id=f'momentum{task}{seed}',accuracy=.65)
                historical[r['trial_id']]=r
    save_json(out/'search_plan.json',dict(stage1=[c.values() for c in first]))
    save_json(out/'stage1_results.json',screen)
    save_json(out/'joint_plan.json',dict(trials=[c.values() for c in joint],stage1_results_sha256=file_hash(out/'stage1_results.json')))
    save_json(out/'search_results.json',search_rows);save_json(out/'review_results.json',dict(rows=[]))
    save_json(out/'formal_results.json',list(finals.values()))
    save_json(out/'review_plan.json',dict(trials=[],search_results_sha256=file_hash(out/'search_results.json')))
    save_json(out/'selected_configs.json',dict(winners=wins,final_trials=[cfg_of_row(r).values() for r in finals.values()],review_results_sha256=file_hash(out/'review_results.json')))
    save_json(old/'formal_results.json',list(historical.values()));save_json(old/'selected_configs.json',dict(winners=old_wins))
    monkeypatch.setattr(report,'RESULTS',out);monkeypatch.setattr(report,'ROOT',history)
    for name,filename in [('FROZEN','search_plan.json'),('SELECTED','selected_configs.json'),('REVIEW','review_plan.json')]:monkeypatch.setattr(report,name,out/filename)
    monkeypatch.setattr(report,'load_frozen',lambda:None);monkeypatch.setattr(report,'completed',lambda c:True)
    monkeypatch.setattr(report,'check_protected',lambda:None);monkeypatch.setattr(report,'sensitivity',lambda *a:[]);monkeypatch.setattr(report,'curves',lambda *a:None)
    report.main()
    for name in ('summary.csv','paired_differences.csv','controlled_differences.csv','eps_sensitivity.csv','final_report.md','final_audit.json'):
        assert (out/name).exists()
    text=(out/'final_report.md').read_text()
    assert '本次不是新盲测' in text and '正式固定 lr/R 对照' in text
    assert json.loads((out/'final_audit.json').read_text())['status']=='passed'
    assert 'fixed_lr_R_eps_only_historical' in (out/'paired_differences.csv').read_text()


def cfg_of_row(row):return Trial(**{k:row[k] for k in Trial.__dataclass_fields__})
