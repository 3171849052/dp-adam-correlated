import copy
import math
from dataclasses import replace
import numpy as np
import pytest
import torch
from exp15c import ROOT,RESULTS
from exp15c.config import Trial,METHODS,FINAL_SEEDS,SEARCH_SEED,save_json,file_hash
from exp15c.optimizer import Momentum,make_optimizer,LogicalBatch,unclipped_microbatch
from exp15c.bandinvmf import momentum_workload,native_matrices,inverse_strategy,materialize,workload_error
from exp15c.privacy import calibrate,epsilon_from_mu,fixed_epoch_sensitivity
from exp15c.noise import BandInvMFNoise
from exp15c.support import historical_matrices,read


@pytest.mark.parametrize('method',METHODS[:2])
def test_unclipped_accumulation_matches_standard_optimizer(method):
    torch.manual_seed(11)
    model=torch.nn.Linear(3,2);ref=copy.deepcopy(model)
    cfg=Trial('cv',method,.03)
    optimizer=make_optimizer(model,cfg)
    reference=torch.optim.Adam(ref.parameters(),lr=.03,betas=(.9,.999),eps=1e-8) if method=='adam-no-noise' else torch.optim.SGD(ref.parameters(),lr=.03*(1-.9),momentum=.9)
    logical=LogicalBatch(model,optimizer,2,logical_size=8)
    x=30*torch.randn(8,3);y=torch.arange(8)%2
    for step in range(3):
        reference.zero_grad();torch.nn.functional.cross_entropy(ref(x),y).backward();reference.step()
        for j in range(2):
            logical.begin_microbatch();unclipped_microbatch(model,x[4*j:4*j+4],y[4*j:4*j+4])
            assert logical.finish_microbatch()==(j==1)
        for a,b in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    assert logical.optimizer_steps==3 and logical.noise is None


def test_momentum_bit_exact_to_exp15b():
    namespace=dict(torch=torch,math=math,METHODS=('momentum',))
    source=(ROOT/'exp15b/optimizer.py').read_text()
    exec(source[source.index('class Optimizer'):],namespace)
    p=torch.nn.Parameter(torch.ones(4));q=torch.nn.Parameter(p.detach().clone())
    a=Momentum([p],.3);b=namespace['Optimizer']([q],'momentum',.3)
    for step in range(5):
        p.grad=q.grad=torch.tensor([.2,-.4,.7,step/10])
        b.freeze_scale();a.step();b.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        torch.testing.assert_close(a.state[p]['exp_avg'],b.state[q]['exp_avg'],rtol=0,atol=0)


@pytest.mark.parametrize('n',[225,250,310])
def test_workload_strategy_privacy(n):
    d,S,W=native_matrices(n);D=materialize(d,n)
    np.testing.assert_allclose(W,momentum_workload(n),rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(S@D,np.eye(n),atol=1e-12)
    rng=np.random.default_rng(3);g=rng.normal(size=n);m=x=0.;trajectory=[]
    for v in g:
        m=.9*m+.1*v;x+=m;trajectory.append(x)
    np.testing.assert_allclose(.1*W@g,trajectory,atol=1e-12,rtol=1e-12)
    for clip in (.1,1.,40.):
        cfg=Trial('nlp' if n==310 else 'cv','momentum-mf-native',.01,clip,smoke_phase='final' if n==250 else 'search')
        p=calibrate(S,cfg.protocol())
        assert epsilon_from_mu(clip*p['sensitivity']/p['innovation_std_sum'],1e-5)==pytest.approx(8.,abs=1e-7)
        old_d,old_S,_,_=historical_matrices(cfg)
        assert fixed_epoch_sensitivity(S,5,n//5)**2*workload_error(W,d)<fixed_epoch_sensitivity(old_S,5,n//5)**2*workload_error(W,old_d)
    assert fixed_epoch_sensitivity(S,5,n//5,steps=2)<fixed_epoch_sensitivity(S,5,n//5)


def test_noise_fir_covariance_and_determinism():
    d,_,_=native_matrices(225);p=torch.zeros(50000)
    a=BandInvMFNoise([p],d,1.,6,7);b=BandInvMFNoise([p],d,1.,6,7)
    draws=[]
    for _ in range(6):
        x=a.next()[0];torch.testing.assert_close(x,b.next()[0],rtol=0,atol=0);draws.append(x.numpy())
    D=materialize(d,225)
    np.testing.assert_allclose(np.cov(draws),(D@D.T)[:6,:6],atol=.04,rtol=.05)


def test_grid_exact_history_and_no_forbidden_methods():
    from exp15c.search import extension_trials
    jobs=extension_trials();assert len(jobs)==63
    old=read(ROOT/'exp15b/results/extended_grid.json')['candidates'].values()
    assert set(c.method for c in jobs)==set(METHODS)
    for task,n in [('cv',17),('nlp',22)]:
        selected=[c for c in jobs if c.task==task and c.private];assert len(selected)==n
        assert {(c.lr,c.clip) for c in selected}=={(r['lr'],r['clip']) for r in old if r['task']==task and r['method']=='momentum'}
    with pytest.raises(ValueError):Trial('cv','adam-no-noise',.001,1.)
    with pytest.raises(ValueError):Trial('cv','momentum-bias-no-noise',.001)


def test_three_seed_selection_and_dedup(monkeypatch):
    from exp15c import search
    jobs=search.extension_trials();schemes={s:{} for s in ('A','B')};preferred={}
    for task in ('cv','nlp'):
        for method in METHODS:
            group=[c for c in jobs if c.task==task and c.method==method]
            fixed=[c for c in group if not c.private or c.clip==1.]
            key=f'{task}:{method}';preferred[key]=[c.id for c in fixed[:2]]
            schemes['A'][key]=[c.id for c in fixed];schemes['B'][key]=[c.id for c in group]
    directory=RESULTS/'unit_selection';directory.mkdir(exist_ok=True)
    path=directory/'grid.json';save_json(path,{})
    for name,file in [('RESULTS',directory),('FROZEN',path),('REVIEW',directory/'review_plan.json'),('SELECTED',directory/'selected.json')]:monkeypatch.setattr(search,name,file)
    monkeypatch.setattr(search,'load_frozen',lambda:dict(candidates={c.id:c.values() for c in jobs},schemes=schemes))
    def run(name,trials):
        rows=[]
        for c in trials:
            origin=replace(c,stage='search',seed=SEARCH_SEED);ids=preferred[f'{c.task}:{c.method}']
            index=ids.index(origin.id) if origin.id in ids else 2
            accuracy=([.8,.7,.01] if name=='search' else [.1,.9,.01])[index]
            rows.append(dict(**c.values(),trial_id=c.id,status='completed',completed_epochs=5,accuracy=accuracy,validation_loss=1-accuracy,official_evaluation_used=False))
        return rows
    monkeypatch.setattr(search,'run_batch',run)
    from exp15c import pipeline
    monkeypatch.setattr(pipeline,'status',lambda *a,**k:None)
    finals=search.search_and_review()
    assert len(finals)==60
    for w in read(search.SELECTED)['winners']:
        assert w['trial_id']==preferred[f"{w['task']}:{w['method']}"][1]
        assert w['accuracy_mean']==pytest.approx((.7+.9+.9)/3)
    assert all(c.seed in FINAL_SEEDS for c in finals)


def test_statistics_and_pairing():
    from exp15c.report import stats,assert_paired
    s=stats([1,2,3]);assert s['mean']==2 and s['sample_std']==1 and s['ci95_low']<2<s['ci95_high']
    a=dict(task='cv',seed=1,initialization_sha256='a',classifier_initialization_sha256='b',train_order_sha256='c',rng_trace=['r'])
    assert_paired(a,dict(a))
    with pytest.raises(ValueError):assert_paired(a,dict(a,rng_trace=['changed']))


def test_report_complete_and_missing_review_rejected(monkeypatch,tmp_path):
    from exp15c import report
    directory=tmp_path/'report';directory.mkdir()
    monkeypatch.setattr(report,'RESULTS',directory)
    def row(c):
        return dict(**c.values(),trial_id=c.id,status='completed',completed_epochs=5,actual_epsilon=8. if c.private else None,
            accuracy=.7,validation_loss=.8,initialization_sha256='init',classifier_initialization_sha256='head',
            train_order_sha256='order',rng_trace=['paired'],official_evaluation_used=c.stage=='final',
            evaluation_scope='official evaluation' if c.stage=='final' else 'internal validation')
    search=[];review=[];formal=[];winners=[]
    for task in ('cv','nlp'):
        for method in METHODS:
            c=Trial(task,method,.001,1. if method=='momentum-mf-native' else None,stage='search',seed=SEARCH_SEED)
            search.append(row(c));review += [row(replace(c,stage='review',seed=seed)) for seed in (20261102,20261103)]
            formal += [row(replace(c,stage='final',seed=seed)) for seed in FINAL_SEEDS]
            winners += [dict(scheme=s,task=task,method=method,lr=c.lr,clip=c.clip) for s in ('A','B')]
    frozen=dict(candidates={r['trial_id']:report.cfg_of(r).values() for r in search})
    monkeypatch.setattr(report,'load_frozen',lambda:frozen)
    monkeypatch.setattr(report,'completed',lambda c:True)
    monkeypatch.setattr(report,'diagnostics',lambda r:(.1,1e-5) if r['clip'] else (None,None))
    monkeypatch.setattr(report,'protected_snapshot',lambda:{})
    history={stage:[dict(r,method='momentum') for r in rows if r['method']=='momentum-mf-native'] for stage,rows in [('search',search),('review',review),('formal',formal)]}
    monkeypatch.setattr(report,'historical_rows',lambda stage:history[stage])
    old_winners=[dict(w,method='momentum') for w in winners if w['method']=='momentum-mf-native']
    original_read=report.read
    monkeypatch.setattr(report,'read',lambda p:dict(winners=old_winners) if p==ROOT/'exp15b/results/selected_configs.json' else original_read(p))
    for name,value in [('extended_grid',frozen),('search_results',search),('review_results',dict(rows=review)),('formal_results',formal),
                       ('review_plan',dict(trials=[report.cfg_of(r).values() for r in review])),('protected_before',{})]:
        save_json(directory/f'{name}.json',value)
    save_json(directory/'selected_configs.json',dict(winners=winners,final_trials=[report.cfg_of(r).values() for r in formal],
        grid_sha256=file_hash(directory/'extended_grid.json'),review_results_sha256=file_hash(directory/'review_results.json')))
    (directory/'mf_comparison.csv').write_text((RESULTS/'mf_comparison.csv').read_text())
    report.main()
    assert original_read(directory/'final_audit.json')['status']=='passed'
    assert (directory/'paired_differences.csv').is_file()
    assert '新盲测' in (directory/'final_report.md').read_text()
    save_json(directory/'review_plan.json',dict(trials=[report.cfg_of(r).values() for r in review[1:]]))
    with pytest.raises(ValueError,match='Missing planned training'):report.main()


def test_historical_artifact_paths_do_not_construct_new_trials():
    from exp15c.report import artifact_dir
    r=next(r for r in read(ROOT/'exp15b/results/search_results.json') if r['method']=='momentum' and not r.get('reuse'))
    assert (artifact_dir(r)/'summary.json').is_file()
    reused=next(r for r in read(ROOT/'exp15b/results/search_results.json') if r['method']=='momentum' and r.get('reuse'))
    assert (artifact_dir(reused)/'summary.json').is_file()
