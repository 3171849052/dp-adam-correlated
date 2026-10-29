from dataclasses import replace
import numpy as np
import pytest
import torch
from exp16f.config import *
from exp16f.optimizer import make_optimizer,LogicalBatch
from exp16f.matrices import load,audit_arrays,workload
from exp16f.noise import OnlineNoise
from exp16f.privacy import calibrate,epsilon_from_mu
from exp16f.search import additions,centers,boundary,plan_review,select,triplet

@pytest.mark.parametrize('n',[225,250,310])
@pytest.mark.parametrize('work,mech',[('identity','iid'),('sgd','b2lt'),('momentum','b2lt'),('momentum-bias','b2lt')])
def test_matrix_privacy_full_stream(n,work,mech):
    a=load(work,mech,n);audit=audit_arrays(a['A'],a['C'],a['D'])
    np.testing.assert_allclose(a['A'],workload(work,n),rtol=1e-13)
    assert audit['objective']==pytest.approx(a['metadata']['audit']['objective'],rel=1e-9)
    for clip in (.1,1,3,10,20,40,500):
        cfg=Trial('nlp' if n==310 else 'cv','adam3-stp-iid',.001,clip,.05,stage='smoke',smoke_phase='final' if n==250 else 'search')
        cal=calibrate(a['C'],cfg.protocol());assert epsilon_from_mu(clip*cal['sensitivity']/cal['innovation_std_sum'],1e-5)==pytest.approx(8,abs=1e-8)
    p=torch.zeros(7);noise=OnlineNoise([p],a,mech,1.7,n,73);g=torch.Generator().manual_seed(73);xs=[];ys=[]
    for i in range(n):
        xs.append(torch.randn(p.shape,generator=g)*1.7);ys.append(noise.next()[0].clone());torch.manual_seed(i);torch.randn(5)
    torch.testing.assert_close(torch.stack(ys),torch.tensor(a['D'],dtype=torch.float32)@torch.stack(xs),rtol=3e-5,atol=3e-5)
    with pytest.raises(ValueError):noise.next()


@pytest.mark.parametrize('method',METHODS)
def test_stp_literal_formula(method):
    model=torch.nn.Linear(2,1,bias=False);p=model.weight
    cfg=Trial('nlp',method,.003,20,.2);opt=make_optimizer(model,cfg)
    class Noise:
        count=0
        def next(self):self.count+=1;return [torch.tensor([[.7,-.4]])]
    noise=Noise();logical=LogicalBatch(model,opt,4,1000,noise,True,.2)
    x=p.detach().clone();m=torch.zeros_like(x);v=m.clone()
    for t in range(1,7):
        scale=1/((v/(1-.999**(t-1))).sqrt()+.2) if t>1 else torch.full_like(v,5.)
        for micro in range(4):
            logical.begin_microbatch();torch.testing.assert_close(p._logical_scale,scale)
            p.grad=torch.tensor([[1.,-2.]])*t
            assert logical.finish_microbatch()==(micro==3)
            assert noise.count==t-1+int(micro==3)
        g=(torch.tensor([[4.,-8.]])*t+torch.tensor([[.7,-.4]])/scale)/1000
        m=.9*m+.1*g;v=.999*v+.001*(m.square() if method.startswith('adam3') else g.square())
        x=x-cfg.lr*(m/(1-.9**t))/((v/(1-.999**t)).sqrt()+1e-8)
        torch.testing.assert_close(p,x);torch.testing.assert_close(p.grad,g)
        torch.testing.assert_close(opt.state[p]['exp_avg_sq'],v)

def test_grid_boundary_selection_and_gpu():
    from exp16f.scheduler import GPUS,capacity
    assert GPUS==(1,2,3) and capacity('nlp',['nlp']) and not capacity('nlp',['nlp','nlp'])
    assert not capacity('cv',['nlp']) and not capacity('nlp',['cv'])
    jobs=additions();assert len(jobs)==len({c.id for c in jobs})==132
    def row(c,a):return dict(**c.values(),trial_id=c.id,status='completed',accuracy=a,validation_loss=1.,official_evaluation_used=False)
    rows=[row(c,c.lr) for c in jobs];ext,evidence=boundary(rows);assert len(ext)==12
    rows += [row(c,c.lr) for c in ext]
    chosen,reviews=plan_review(rows);assert len(reviews)==48
    assert triplet(reviews[0],rows) is None
    rows += [row(c,.5) for c in reviews]
    winners,evidence=select(chosen,rows);assert all(len(ms)==6 for ms in winners.values())
    assert all(len(e['selected']['runs'])==3 for e in evidence.values())
    for c in jobs:
        assert c.scaled and c.total_steps==(225 if c.task=='cv' else 310)
        assert replace(c,stage='final').total_steps==(250 if c.task=='cv' else 310)
    rows[-1]['official_evaluation_used']=True
    with pytest.raises(AssertionError):select(chosen,rows)
