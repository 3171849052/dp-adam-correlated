from dataclasses import replace
import numpy as np
import pytest
import torch
from exp16e import ROOT
from exp16e.config import *
from exp16e.optimizer import make_optimizer,LogicalBatch
from exp16e.matrices import load,audit_arrays,workload
from exp16e.noise import OnlineNoise
from exp16e.privacy import calibrate,epsilon_from_mu
from exp16e.search import additions,NEW_NLP,plan_review,select,triplet
from exp16e.history import key

@pytest.mark.parametrize('method',[ADAM3[0],MOMENTUM[0],STPEX])
def test_literal_update_and_noise_once(method):
    model=torch.nn.Linear(2,1,bias=False);p=model.weight
    cfg=Trial('nlp',method,.003,20,.2 if method==STPEX else None);opt=make_optimizer(model,cfg)
    class Noise:
        count=0
        def next(self):self.count+=1;return [torch.tensor([[.7,-.4]])]
    noise=Noise();logical=LogicalBatch(model,opt,4,1000,noise,cfg.scaled,cfg.eps_scale)
    x=p.detach().clone();m=torch.zeros_like(x);v=m.clone()
    for t in range(1,7):
        d=(v/(1-.999**(t-1))).sqrt()+.2 if t>1 else torch.full_like(v,.2)
        for micro in range(4):
            logical.begin_microbatch();p.grad=torch.tensor([[1.,-2.]])*t
            assert logical.finish_microbatch()==(micro==3)
            assert noise.count==t-1+int(micro==3)
        total=torch.tensor([[4.,-8.]])*t
        g=(total/d+torch.tensor([[.7,-.4]]))/1000 if method==STPEX else (total+torch.tensor([[.7,-.4]]))/1000
        m=.9*m+.1*g
        if method==ADAM3[0]:
            v=.999*v+.001*m.square();x=x-cfg.lr*(m/(1-.9**t))/((v/(1-.999**t)).sqrt()+1e-8)
        elif method==STPEX:
            v=.999*v+.001*(d*g).square();x=x-cfg.lr*m/(1-.9**t)
        else:x=x-cfg.lr*m
        torch.testing.assert_close(p,x)
        torch.testing.assert_close(opt.state[p]['exp_avg'],m)
        if method!=MOMENTUM[0]:torch.testing.assert_close(opt.state[p]['exp_avg_sq'],v)
    assert (ROOT/'exp14a/optimizer.py').read_bytes()==(ROOT/'exp16e/adam3.py').read_bytes()
    assert (ROOT/'exp15c/optimizer.py').read_bytes()==(ROOT/'exp16e/momentum_optimizer.py').read_bytes()

@pytest.mark.parametrize('n',[225,250,310])
@pytest.mark.parametrize('work,mech',[('identity','iid'),('sgd','b2lt'),('momentum','b2lt'),('momentum-bias','b2lt')])
def test_matrix_privacy_full_stream(n,work,mech):
    a=load(work,mech,n);audit=audit_arrays(a['A'],a['C'],a['D'])
    np.testing.assert_allclose(a['A'],workload(work,n),rtol=1e-13)
    assert audit['objective']==pytest.approx(a['metadata']['audit']['objective'],rel=1e-9)
    for clip in (.1,1,3,10,20,40,500):
        cfg=Trial('nlp' if n==310 else 'cv','adam-iid',.001,clip,stage='smoke',smoke_phase='final' if n==250 else 'search')
        cal=calibrate(a['C'],cfg.protocol());assert epsilon_from_mu(clip*cal['sensitivity']/cal['innovation_std_sum'],1e-5)==pytest.approx(8,abs=1e-8)
    p=torch.zeros(7);noise=OnlineNoise([p],a,mech,1.7,n,73);g=torch.Generator().manual_seed(73);xs=[];ys=[]
    for i in range(n):
        xs.append(torch.randn(p.shape,generator=g)*1.7);ys.append(noise.next()[0].clone());torch.manual_seed(i);torch.randn(5)
    torch.testing.assert_close(torch.stack(ys),torch.tensor(a['D'],dtype=torch.float32)@torch.stack(xs),rtol=3e-5,atol=3e-5)
    with pytest.raises(ValueError):noise.next()

def test_grid_and_gpu_policy():
    from exp16e.scheduler import GPUS,capacity,runnable
    jobs=additions();assert len(jobs)==67 and sum(c.task=='cv' for c in jobs)==9
    assert sum(c.method==STPEX for c in jobs)==8
    assert sum(c.method.startswith('momentum') and c.task=='nlp' for c in jobs)==21
    assert sum(c.method.startswith('adam3') and c.task=='nlp' for c in jobs)==20
    assert GPUS==(1,2,3) and capacity('nlp',[]) and not capacity('nlp',['nlp'])
    active={i:(Trial('nlp',ADAM[0],.1,1),None,i,None) for i in (1,2)}
    assert not runnable('nlp',active) and runnable('cv',active)
    for c in jobs:
        assert c.total_steps==(225 if c.task=='cv' else 310)
        assert replace(c,stage='final').total_steps==(250 if c.task=='cv' else 310)
    with pytest.raises(ValueError):Trial('nlp','adam-mf-sgd-bandinvmf',.01,10)

def test_selection_requires_three_internal_seeds():
    c=Trial('nlp',MOMENTUM[0],.1,.3)
    def row(c,a):return dict(**c.values(),trial_id=c.id,status='completed',accuracy=a,validation_loss=1.,official_evaluation_used=False)
    rows=[row(c,.9),row(replace(c,stage='review',seed=20261102),.7)]
    assert triplet(c,rows) is None
    rows.append(row(replace(c,stage='review',seed=20261103),.5));assert triplet(c,rows)['mean_accuracy']==pytest.approx(.7)
    rows[-1]['official_evaluation_used']=True
    with pytest.raises(AssertionError):triplet(c,rows)

def test_report_stats_and_pair_direction():
    from exp16e.report import stats
    s=stats(range(10));assert s['mean']==4.5 and s['n']==10 and s['ci95_low']<4.5<s['ci95_high']
    assert stats([.1]*10)['sd']==0

def test_full_selection_plan_without_official_results():
    from exp16e.history import incumbents,config
    old=incumbents();rows=[]
    def append(c,a):rows.append(dict(**c.values(),trial_id=c.id,status='completed',accuracy=a,validation_loss=1.,official_evaluation_used=False))
    for methods in old.values():
        for v in methods.values():
            c=config(v);append(c,.7)
            for seed in RECHECK_SEEDS:append(replace(c,stage='review',seed=seed),.7)
    for c in additions():append(c,.8)
    chosen,decisions,jobs=plan_review(rows)
    assert len(jobs)==48
    for c in jobs:append(c,.8)
    winners,evidence=select(chosen,decisions,rows)
    assert len(winners['cv'])==len(winners['nlp'])==13
    assert all(len(e['selected']['runs'])==3 for e in evidence.values())
    for r in rows:
        if r['stage']=='search' and key(config(r)) in {key(c) for c in additions()}:r['accuracy']=.6
    chosen,decisions,jobs=plan_review(rows)
    assert len(jobs)==32

def test_historical_selection_pool_excludes_controlled_reviews():
    import json
    ids={r['trial_id'] for r in json.loads((ROOT/'exp16d/results/review_results.json').read_text())}
    extra=[]
    for p in (ROOT/'exp16d/results/review/cv').glob('*/summary.json'):
        r=json.loads(p.read_text())
        if r['trial_id'] not in ids:extra.append(r['trial_id'])
    assert len(extra)==2
