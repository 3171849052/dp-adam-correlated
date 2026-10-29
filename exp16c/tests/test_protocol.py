import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from exp16c.config import *
from exp16c.search import additions,select
from exp16c.optimizer import make_optimizer,LogicalBatch
from exp16c.clipping import StandardGhostModule,clipped_microbatch
from exp16c.matrices import load,audit_arrays,workload
from exp16c.noise import OnlineNoise
from exp16c.privacy import calibrate,epsilon_from_mu

@pytest.mark.parametrize('method',MOMENTUM_METHODS)
def test_momentum_literal_and_noise_once(method):
    model=torch.nn.Linear(2,1,bias=False);p=model.weight;cfg=Trial('cv',method,.5,3.)
    opt=make_optimizer(model,cfg)
    class Noise:
        count=0
        def next(self):self.count+=1;return [torch.tensor([[.7,-.4]])]
    noise=Noise();logical=LogicalBatch(model,opt,4,1000,noise)
    x=p.detach().clone();m=torch.zeros_like(x)
    for step in range(4):
        for micro in range(4):
            logical.begin_microbatch();p.grad=torch.tensor([[1.,-2.]])
            assert logical.finish_microbatch()==(micro==3)
            assert noise.count==step+int(micro==3)
        g=(torch.tensor([[4.,-8.]])+torch.tensor([[.7,-.4]]))/1000
        m=.9*m+(1-.9)*g;x=x-.5*m
        torch.testing.assert_close(p,x,rtol=0,atol=0);torch.testing.assert_close(opt.state[p]['exp_avg'],m,rtol=0,atol=0)
    assert len(opt.state[p])==1

@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_clipped_accumulation_matches_per_example(device):
    torch.manual_seed(4);base=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Tanh(),torch.nn.Linear(5,2)).to(device)
    model=StandardGhostModule(copy.deepcopy(base),.3);x=torch.randn(8,3,device=device);y=torch.arange(8,device=device)%2
    expected=[torch.zeros_like(p) for p in base.parameters()]
    for i in range(8):
        base.zero_grad();torch.nn.functional.cross_entropy(base(x[i:i+1]),y[i:i+1]).backward()
        norm=sum(p.grad.square().sum() for p in base.parameters()).sqrt()
        for s,p in zip(expected,base.parameters()):s.add_(p.grad*min(1.,.3/norm.item()))
    cfg=Trial('cv','momentum-iid',1.,.3);opt=make_optimizer(model,cfg);logical=LogicalBatch(model,opt,4,8)
    for i in range(0,8,2):
        logical.begin_microbatch();clipped_microbatch(model,x[i:i+2],y[i:i+2],.3);logical.finish_microbatch()
    for p,s in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,s/8,rtol=3e-5,atol=1e-7)
    model.remove_hooks()

@pytest.mark.parametrize('n',[225,250])
@pytest.mark.parametrize('work,mech',[('identity','iid'),('sgd','b2lt'),('momentum','b2lt'),('momentum-bias','b2lt')])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_matrix_privacy_full_stream(n,work,mech,device):
    a=load(work,mech,n);audit=audit_arrays(a['A'],a['C'],a['D'])
    np.testing.assert_allclose(a['A'],workload(work,n),rtol=1e-13)
    assert audit['objective']==pytest.approx(a['metadata']['audit']['objective'],rel=1e-9)
    for clip in (.3,1.,3.,12.,24.):
        cfg=Trial('cv','momentum-iid',.5,clip,stage='smoke',smoke_phase='final' if n==250 else 'search');cal=calibrate(a['C'],cfg.protocol())
        assert epsilon_from_mu(clip*cal['sensitivity']/cal['innovation_std_sum'],1e-5)==pytest.approx(8,abs=1e-8)
    p=torch.zeros(7,device=device);noise=OnlineNoise([p],a,mech,1.7,n,73);g=torch.Generator(device=device).manual_seed(73);xs=[];ys=[]
    for i in range(n):
        xs.append(torch.randn(p.shape,device=device,generator=g)*1.7);ys.append(noise.next()[0].clone());torch.manual_seed(i);torch.randn(5)
    torch.testing.assert_close(torch.stack(ys),torch.tensor(a['D'],device=device,dtype=torch.float32)@torch.stack(xs),rtol=3e-5,atol=3e-5)
    with pytest.raises(ValueError):noise.next()

@pytest.mark.parametrize('work',['sgd','momentum','momentum-bias'])
def test_empirical_covariance(work):
    a=load(work,'b2lt',225);noise=OnlineNoise([torch.zeros(16000)],a,'b2lt',1.,225,81);indices=[0,1,4,44,45,224];ys=[]
    for i in range(225):
        y=noise.next()[0]
        if i in indices:ys.append(y.numpy())
    target=(a['D']@a['D'].T)[np.ix_(indices,indices)]
    assert np.linalg.norm(np.cov(ys)-target)/np.linalg.norm(target)<.055

def test_grid_and_protocol():
    from exp16c.scheduler import capacity
    jobs=additions();assert len(jobs)==len({c.id for c in jobs})==25
    assert all(c.task=='cv' and c.total_steps==225 and c.physical_batch==250 for c in jobs)
    assert all(replace(c,stage='final',seed=FINAL_SEEDS[0]).total_steps==250 for c in jobs)
    assert capacity('cv',[]) and not capacity('cv',['cv']) and not capacity('nlp',[])
    with pytest.raises(ValueError):Trial('nlp','momentum-iid',1.,1.)
    assert Trial('cv','momentum-mf-momentum-b2lt',1.,1.).workload=='momentum'

from exp16c.stpex_optimizer import Optimizer, LogicalBatch as StpexLogicalBatch
@pytest.mark.parametrize('method',[STPEX])
def test_literal_stpex_and_microbatch_noise(method):
    model=torch.nn.Linear(2,1,bias=False);p=model.weight
    opt=Optimizer(model.parameters(),method,.3,eps_scale=.2)
    class Noise:
        count=0
        def next(self):
            self.count+=1
            return [torch.tensor([[.7,-.4]])]
    noise=Noise();logical=StpexLogicalBatch(model,opt,4,1000,noise,True,.2)
    x=p.detach().clone();u=torch.zeros_like(x);v=u.clone()
    for t in range(1,5):
        d=(v/(1-.999**(t-1))).sqrt()+.2 if t>1 else torch.full_like(v,.2)
        for micro in range(4):
            logical.begin_microbatch();torch.testing.assert_close(p._logical_denominator,d)
            p.grad=torch.tensor([[1.,-2.]])
            assert logical.finish_microbatch()==(micro==3)
            assert noise.count==t-1+int(micro==3)
        q=(torch.tensor([[4.,-8.]])/d+torch.tensor([[.7,-.4]]))/1000
        u=.9*u+.1*q;v=.999*v+.001*(d*q)**2;x=x-.3*u/(1-.9**t)
        torch.testing.assert_close(p,x);torch.testing.assert_close(opt.state[p]['exp_avg'],u)
        torch.testing.assert_close(opt.state[p]['exp_avg_sq'],v)
    p.grad=torch.full_like(p,float('nan'));before=p.clone()
    with pytest.raises(FloatingPointError):opt.step()
    torch.testing.assert_close(p,before)

