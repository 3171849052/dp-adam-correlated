import itertools
import numpy as np
import pytest
import torch
from exp16b.config import Trial,METHODS
from exp16b.optimizer import Optimizer,LogicalBatch
from exp16b.matrices import load,audit
from exp16b.noise import OnlineNoise
from exp16b.privacy import calibrate,epsilon_from_mu,fixed_epoch_sensitivity
from exp16b.search import coarse_jobs

@pytest.mark.parametrize('method',METHODS)
def test_literal_stpex_and_microbatch_noise(method):
    model=torch.nn.Linear(2,1,bias=False);p=model.weight
    opt=Optimizer(model.parameters(),method,.3,eps_scale=.2)
    class Noise:
        count=0
        def next(self):
            self.count+=1
            return [torch.tensor([[.7,-.4]])]
    noise=Noise();logical=LogicalBatch(model,opt,4,1000,noise,True,.2)
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

@pytest.mark.parametrize('n',[225,310])
@pytest.mark.parametrize('mech',['bandinvmf','b2lt'])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_matrix_privacy_and_full_stream(n,mech,device):
    a=load('momentum-bias',mech,n);audit(a);C,D=a['C'],a['D']
    patterns=np.array(list(itertools.product((-1.,0.,1.),repeat=5))).T
    direct=max(np.linalg.norm(C[:,j+n//5*np.arange(5)]@patterns,axis=0).max() for j in range(n//5))
    assert direct<=fixed_epoch_sensitivity(C,5,n//5)*(1+1e-10)
    for clip in (1.,3.,5.,10.,20.,40.):
        cfg=Trial('nlp' if n==310 else 'cv','stpex-mf-momentum-bias-'+mech,.15,clip,.1)
        cal=calibrate(C,cfg.protocol())
        assert epsilon_from_mu(clip*cal['sensitivity']/cal['innovation_std_sum'],1e-5)==pytest.approx(8,abs=1e-8)
    p=torch.zeros(3,device=device);noise=OnlineNoise([p],a,mech,1.7,n,73)
    gen=torch.Generator(device=device).manual_seed(73);xs=[];ys=[]
    for t in range(n):
        xs.append(torch.randn(p.shape,device=device,generator=gen)*1.7);ys.append(noise.next()[0].clone())
        torch.manual_seed(t);torch.randn(11)
    torch.testing.assert_close(torch.stack(ys),torch.tensor(D,device=device,dtype=torch.float32)@torch.stack(xs),atol=3e-5,rtol=3e-5)
    with pytest.raises(ValueError):noise.next()

@pytest.mark.parametrize('mech',['bandinvmf','b2lt'])
def test_empirical_covariance(mech):
    a=load('momentum-bias',mech,225);noise=OnlineNoise([torch.zeros(16000)],a,mech,1.,225,81)
    indices=[0,1,4,44,45,224];samples=[]
    for t in range(225):
        y=noise.next()[0]
        if t in indices:samples.append(y.numpy())
    predicted=(a['D']@a['D'].T)[np.ix_(indices,indices)]
    assert np.linalg.norm(np.cov(samples)-predicted)/np.linalg.norm(predicted)<.055

def test_grid_and_gpu_policy():
    from exp16b.scheduler import capacity
    jobs=coarse_jobs();assert len(jobs)==len({c.id for c in jobs})==108
    assert all(c.seed==20261101 and c.stage=='search' and c.total_steps in (225,310) for c in jobs)
    assert capacity('nlp',['nlp']) and not capacity('nlp',['nlp','nlp'])
    assert not capacity('cv',['nlp']) and not capacity('nlp',['cv']) and not capacity('cv',['cv'])
    with pytest.raises(ValueError):Trial('cv',METHODS[0],.15,1.,.1,stage='final')
