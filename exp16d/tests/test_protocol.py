import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from exp16d import ROOT,RESULTS
from exp16d.config import *
from exp16d.search import additions,controlled,top_two,select
from exp16d.optimizer import make_optimizer,LogicalBatch
from exp16d.clipping import StandardGhostModule,clipped_microbatch
from exp16d.matrices import load,audit_arrays,workload
from exp16d.noise import OnlineNoise
from exp16d.privacy import calibrate,epsilon_from_mu

@pytest.mark.parametrize('method',METHODS)
def test_literal_adam3_and_noise_once(method):
    model=torch.nn.Linear(2,1,bias=False);p=model.weight;cfg=Trial('cv',method,.0004,20)
    opt=make_optimizer(model,cfg)
    class Noise:
        count=0
        def next(self):self.count+=1;return [torch.tensor([[.7,-.4]])]
    noise=Noise();logical=LogicalBatch(model,opt,4,1000,noise)
    x=p.detach().clone();m=torch.zeros_like(x);v=m.clone()
    for t in range(1,7):
        for micro in range(4):
            logical.begin_microbatch();p.grad=torch.tensor([[1.,-2.]])*t
            assert logical.finish_microbatch()==(micro==3)
            assert noise.count==t-1+int(micro==3)
        g=(torch.tensor([[4.,-8.]])*t+torch.tensor([[.7,-.4]]))/1000
        m=.9*m+.1*g;v=.999*v+.001*m.square();x=x-cfg.lr*(m/(1-.9**t))/((v/(1-.999**t)).sqrt()+1e-8)
        torch.testing.assert_close(p,x);torch.testing.assert_close(opt.state[p]['exp_avg_sq'],v)
    assert (ROOT/'exp14a/optimizer.py').read_bytes()==(ROOT/'exp16d/adam3.py').read_bytes()

def test_clipping_reference():
    torch.manual_seed(4);base=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Tanh(),torch.nn.Linear(5,2))
    model=StandardGhostModule(copy.deepcopy(base),.3);x=torch.randn(8,3);y=torch.arange(8)%2
    expected=[torch.zeros_like(p) for p in base.parameters()]
    for i in range(8):
        base.zero_grad();torch.nn.functional.cross_entropy(base(x[i:i+1]),y[i:i+1]).backward()
        norm=sum(p.grad.square().sum() for p in base.parameters()).sqrt()
        for s,p in zip(expected,base.parameters()):s.add_(p.grad*min(1.,.3/norm.item()))
    cfg=Trial('cv',METHODS[0],.0004,.3);logical=LogicalBatch(model,make_optimizer(model,cfg),4,8)
    for i in range(0,8,2):
        logical.begin_microbatch();clipped_microbatch(model,x[i:i+2],y[i:i+2],.3);logical.finish_microbatch()
    for p,s in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,s/8,rtol=3e-5,atol=1e-7)
    model.remove_hooks()

@pytest.mark.parametrize('n',[225,250])
@pytest.mark.parametrize('work,mech',[('identity','iid'),('sgd','b2lt'),('momentum','b2lt'),('momentum-bias','b2lt')])
def test_matrix_privacy_full_stream(n,work,mech):
    a=load(work,mech,n);audit=audit_arrays(a['A'],a['C'],a['D'])
    np.testing.assert_allclose(a['A'],workload(work,n),rtol=1e-13)
    assert audit['objective']==pytest.approx(a['metadata']['audit']['objective'],rel=1e-9)
    for clip in (10,20,30,40,60):
        cfg=Trial('cv',METHODS[0],.0004,clip,stage='smoke',smoke_phase='final' if n==250 else 'search');cal=calibrate(a['C'],cfg.protocol())
        assert epsilon_from_mu(clip*cal['sensitivity']/cal['innovation_std_sum'],1e-5)==pytest.approx(8,abs=1e-8)
    p=torch.zeros(7);noise=OnlineNoise([p],a,mech,1.7,n,73);g=torch.Generator().manual_seed(73);xs=[];ys=[]
    for i in range(n):
        xs.append(torch.randn(p.shape,generator=g)*1.7);ys.append(noise.next()[0].clone());torch.manual_seed(i);torch.randn(5)
    torch.testing.assert_close(torch.stack(ys),torch.tensor(a['D'],dtype=torch.float32)@torch.stack(xs),rtol=3e-5,atol=3e-5)
    with pytest.raises(ValueError):noise.next()

def test_selection_control_reuse_and_protocol():
    from exp16d.scheduler import capacity
    jobs=additions();assert len(jobs)==len({c.id for c in jobs})==20
    assert all(c.total_steps==225 and c.physical_batch==250 for c in jobs)
    assert all(replace(c,stage='final').total_steps==250 for c in jobs)
    assert {c.id for c in controlled() if c.stage=='search'} <= {c.id for c in jobs}
    assert capacity('cv',[]) and not capacity('cv',['cv']) and not capacity('nlp',[])
    rows=[dict(**c.values(),trial_id=c.id,status='completed',accuracy=.5+i*.001,validation_loss=1.,official_evaluation_used=False) for i,c in enumerate(jobs)]
    top=top_two(rows);assert all(len(v)==2 for v in top.values())
    for vs in top.values():
        for v in vs:
            for seed in RECHECK_SEEDS:
                c=replace(Trial(**v),seed=seed,stage='review');rows.append(dict(**c.values(),trial_id=c.id,status='completed',accuracy=.6,validation_loss=1.,official_evaluation_used=False))
    winners,evidence=select(top,rows);assert set(winners['cv'])==set(METHODS)
    assert all(len(v['selected']['trial_ids'])==3 for v in evidence.values())
    rows[0]['official_evaluation_used']=True
    with pytest.raises(ValueError):top_two(rows)

def test_real_pretrained_cpu_smoke(monkeypatch):
    # Reduced CPU diagnostic only; queued GPU gate uses unchanged 250/1000 protocol.
    import builtins
    from exp16d.cv_model import pretrained_vit
    from exp16d.data import cv_datasets
    from exp16d.randomness import seed_all
    original=builtins.open
    def guarded(file,*args,**kw):
        if str(file).endswith('cifar-100-python/test'):raise AssertionError('Official test accessed')
        return original(file,*args,**kw)
    monkeypatch.setattr(builtins,'open',guarded)
    seed_all(SEARCH_SEED);base,meta=pretrained_vit();cfg=Trial('cv',METHODS[0],.0004,20)
    train,valid,order=cv_datasets(cfg,meta);assert len(train)==45000 and len(valid)==5000
    model=StandardGhostModule(base,20);opt=make_optimizer(model,cfg)
    a=load('identity','iid',225);cal=calibrate(a['C'],cfg.protocol());noise=OnlineNoise(model.parameters(),a,'iid',cal['innovation_std_sum'],225,cfg.seed+1)
    logical=LogicalBatch(model,opt,2,2,noise)
    for i in range(2):
        x,y=train[i];logical.begin_microbatch();clipped_microbatch(model,x[None],torch.tensor([y]),20);logical.finish_microbatch()
    assert logical.optimizer_steps==1 and noise.step_count==1
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.remove_hooks()
    save_json(RESULTS/'cpu_smoke.json',dict(status='passed',model='local pretrained ViT-Tiny',diagnostic_batch=2,production_physical_batch=250,production_logical_batch=1000,official_test_access=False))
