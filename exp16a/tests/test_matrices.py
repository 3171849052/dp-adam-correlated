import numpy as np
import pytest
import torch
from exp16a import RESULTS
from exp16a.matrices import load,workload,audit_arrays,materialize,bisr
from exp16a.privacy import fixed_epoch_sensitivity,calibrate,epsilon_from_mu,gdp_delta
from exp16a.noise import OnlineNoise
from exp16a.config import Trial

CELLS=[('identity','iid')]+[(w,m) for w in ('sgd','momentum','momentum-bias') for m in ('bandinvmf','b2lt','dtc')]
@pytest.mark.parametrize('n',[225,250,310])
@pytest.mark.parametrize('work,mech',CELLS)
def test_matrix_objective_privacy_covariance_and_stream(n,work,mech):
    a=load(work,mech,n);A,C,D=a['A'],a['C'],a['D'];meta=a['metadata'];audit=audit_arrays(A,C,D)
    np.testing.assert_allclose(A,workload(work,n),rtol=1e-13)
    independent=np.linalg.inv(C)
    np.testing.assert_allclose(D,independent,atol=1e-9,rtol=1e-9)
    # Exhaust all scalar sign combinations and zero-out subsets independently.
    import itertools
    patterns=np.array(list(itertools.product((-1.,0.,1.),repeat=5))).T
    direct=max(np.linalg.norm(C[:,j+(n//5)*np.arange(5)]@patterns,axis=0).max() for j in range(n//5))
    assert direct<=audit['sensitivity']*(1+1e-10)
    objective=fixed_epoch_sensitivity(C,5,n//5)**2*np.sum((A@independent)**2)/n
    assert objective==pytest.approx(meta['audit']['objective'],rel=1e-9)
    if mech=='iid':assert a['d'].shape==(1,)
    if mech=='bandinvmf':assert np.max(np.abs(np.tril(D,-4)))==0
    if mech=='b2lt':
        theta=np.array(meta['theta']);omega=np.array(meta['omega']);P=meta['P']
        assert P==n//5 and meta['Q']==5
        assert np.all(0<theta) and np.all(theta<1) and np.all(omega<0)
        assert 1+omega.sum()>0 and 1+np.sum(omega/(1-theta))>0
        np.testing.assert_allclose(a['d'][1:],(omega[:,None]*theta[:,None]**np.arange(P-1)).sum(0))
        assert np.max(np.abs(np.tril(D,-P)))==0
    if mech=='dtc':
        assert np.all(np.diff(C[:,0])<=1e-12) and np.min(C)>=-1e-12
        assert objective<=5*np.sum(A*A)/n*(1+1e-10)
    if mech!='iid':
        assert meta['optimizer'].startswith('L-BFGS') and len(meta['trace'])>2
        assert objective<=meta['initial_error']*(1+1e-9)
    for clip in (.1,30.,500.):
        cfg=Trial('nlp' if n==310 else 'cv','adam-iid',.001,clip,stage='smoke',smoke_phase='final' if n==250 else 'search')
        p=calibrate(C,cfg.protocol());mu=clip*audit['sensitivity']/p['innovation_std_sum']
        assert epsilon_from_mu(mu,1e-5)==pytest.approx(8.,abs=1e-8)
        # Independent normal-CDF GDP expression checks the stable log formula.
        from scipy.stats import norm
        assert norm.cdf(-8/mu+mu/2)-np.exp(8)*norm.cdf(-8/mu-mu/2)==pytest.approx(1e-5,abs=1e-12)
    # Full horizon with multiple parameter shapes; independent innovation replay.
    params=[torch.zeros(3),torch.zeros(2,2)];sigma=1.7;noise=OnlineNoise(params,a,mech,sigma,n,73)
    gen=torch.Generator().manual_seed(73);innovations=[];observed=[]
    for t in range(n):
        innovations.append(torch.cat([(torch.randn(p.shape,generator=gen)*sigma).flatten() for p in params]))
        observed.append(torch.cat([x.flatten() for x in noise.next()]))
        if mech=='iid':assert not noise.history
        if mech=='bandinvmf':assert len(noise.history)<=3
        if mech=='b2lt':assert len(noise.buffers)==5
        torch.manual_seed(t);torch.randn(11) # global RNG must not influence DP innovations
    expected=torch.tensor(D,dtype=torch.float32)@torch.stack(innovations)
    torch.testing.assert_close(torch.stack(observed),expected,rtol=3e-5,atol=3e-5)
    assert noise.step_count==n
    with pytest.raises(ValueError):noise.next()
    assert np.allclose(a['covariance_clip1'],audit['sigma_sum_clip1']**2*D@D.T)

@pytest.mark.parametrize('work',['sgd','momentum','momentum-bias'])
def test_literal_workload_definitions(work):
    n=12;expected=np.zeros((n,n))
    for t in range(1,n+1):
        for j in range(1,t+1):
            expected[t-1,j-1]=1 if work=='sgd' else 1-.9**(t-j+1) if work=='momentum' else sum(.9**(s-j)/(1-.9**s) for s in range(j,t+1))
    np.testing.assert_allclose(workload(work,n),expected,rtol=1e-14)
    if work=='momentum-bias':assert not np.allclose(np.diag(expected),expected[0,0])

def test_bisr_initialization():
    np.testing.assert_allclose(bisr(np.ones(4)),[1,-.5,-.125,-.0625],rtol=1e-13)

@pytest.mark.parametrize('mech',['bandinvmf','b2lt','dtc'])
def test_empirical_online_covariance(mech):
    n=225;a=load('sgd',mech,n);indices=[0,1,4,44,45,224]
    noise=OnlineNoise([torch.zeros(16000)],a,mech,1.,n,81);samples=[]
    for t in range(n):
        y=noise.next()[0]
        if t in indices:samples.append(y.numpy())
    predicted=(a['D']@a['D'].T)[np.ix_(indices,indices)]
    measured=np.cov(samples)
    assert np.linalg.norm(measured-predicted)/np.linalg.norm(predicted)<.055

@pytest.mark.parametrize('mech',['b2lt','dtc'])
def test_cuda_replay_full_horizon(mech):
    n=225;a=load('sgd',mech,n);p=torch.zeros(3,device='cuda:0');noise=OnlineNoise([p],a,mech,1.,n,83)
    gen=torch.Generator(device='cuda:0').manual_seed(83);xi=[];ys=[]
    for _ in range(n):xi.append(torch.randn(p.shape,device=p.device,generator=gen));ys.append(noise.next()[0].clone())
    torch.testing.assert_close(torch.stack(ys),torch.tensor(a['D'],device=p.device,dtype=torch.float32)@torch.stack(xi),atol=2e-5,rtol=2e-5)
