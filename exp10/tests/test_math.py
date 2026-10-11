import numpy as np
import pytest
import torch
from exp10.matrices import get_matrix,components,optimize,weights
from exp10.bandinvmf import bias_matrices,momentum_bias_workload,workload_error
from exp10.noise import materialize,BandInvMFNoise
from exp10.privacy import calibrate,epsilon_from_mu,fixed_epoch_sensitivity
from exp10.config import Trial,METHODS,LAMBDAS

@pytest.mark.parametrize('n',[225,250,310])
@pytest.mark.parametrize('lam',[None,*LAMBDAS])
def test_inverse_true_sensitivity_gdp_and_objectives(n,lam):
    d,S,W,meta=get_matrix(n,lam)
    np.testing.assert_allclose(materialize(d,n)@S,np.eye(n),atol=3e-12)
    assert d[0]==1 and len(d)==4
    assert fixed_epoch_sensitivity(S,5,n//5)==pytest.approx(meta['sensitivity'],rel=1e-12)
    # Independent full Gram bound and complete W1 objective.
    gram=S.T@S
    sensitivity=max(np.abs(gram[np.ix_(idx,idx)]).sum() for j in range(n//5)
                    for idx in [j+(n//5)*np.arange(5)])**.5
    assert sensitivity==pytest.approx(meta['sensitivity'],rel=1e-12)
    assert meta['J1']==pytest.approx(sensitivity**2*np.sum((W@materialize(d,n))**2)/n,rel=1e-12)
    cfg=Trial('nlp' if n==310 else 'cv',METHODS[0],.003,30,.3,stage='smoke',smoke_phase='final' if n==250 else 'search')
    for C in (1.,20.,30.,100.):
        protocol=cfg.protocol();protocol['privacy']['max_grad_norm']=C;p=calibrate(S,protocol)
        assert epsilon_from_mu(C*sensitivity/p['innovation_std_sum'],1e-5)==pytest.approx(8.,abs=1e-8)
    if lam is None:
        np.testing.assert_array_equal(d,bias_matrices(n)[0])
    else:
        assert meta['normalized_objective']<=1+lam+1e-8
        assert meta['normalized_objective']==pytest.approx(meta['J1']/meta['baseline']['J1']+lam*meta['J2']/meta['baseline']['J2'],rel=1e-10)
@pytest.mark.parametrize('n',[225,250,310])
def test_lambda_zero_consistency(n):
    d,meta=optimize(n,0.)
    original=bias_matrices(n)
    assert meta['J1']==pytest.approx(original[3]['optimized_objective'],rel=1e-8)
    np.testing.assert_allclose(d,original[0],atol=2e-6,rtol=2e-6)
def test_complete_workload_and_second_moment_weights():
    n=12;expected=np.zeros((n,n))
    for t in range(1,n+1):
        for s in range(1,t+1): expected[t-1,s-1]=sum(.9**(u-s)/(1-.9**u) for u in range(s,t+1))
    np.testing.assert_allclose(momentum_bias_workload(n,.9),expected,rtol=1e-14)
    w=weights(n);np.testing.assert_allclose(w.sum(axis=1),1.,atol=2e-14)
    assert np.count_nonzero(np.triu(w,1))==0
@pytest.mark.parametrize('n',[225,250,310])
def test_gaussian_J2_monte_carlo(n):
    d,S,W,meta=get_matrix(n,.3);D=materialize(d,n);sigma=meta['sigma_avg_C1'];w=weights(n)
    # Independent Gaussian squared-observation simulation (Isserlis identity).
    Z=np.random.default_rng(813+n).standard_normal((n,16000))
    noise=sigma*(D@Z);second=w@(noise**2)
    empirical=second.var(axis=1,ddof=1).mean()
    assert empirical==pytest.approx(meta['J2'],rel=.065)
    covariance=sigma**2*(D@D.T)
    direct=2*np.mean([row@(covariance*covariance)@row for row in w])
    assert direct==pytest.approx(meta['J2'],rel=1e-12)
    assert components(d,n,W)['J2']==pytest.approx(meta['J2'],rel=1e-12)
def test_full_trajectory_innovations_and_independent_rng():
    params=[torch.zeros(3),torch.zeros(2,2)];d=get_matrix(225,.3)[0]
    noise=BandInvMFNoise(params,d,2.,225,46);gen=torch.Generator().manual_seed(46);history=[]
    for step in range(225):
        history.insert(0,[torch.randn(p.shape,generator=gen)*2. for p in params]);actual=noise.next()
        torch.manual_seed(123+step);torch.randn(100)
        for j in range(2):
            expected=sum(float(d[lag])*history[lag][j] for lag in range(min(4,step+1)))
            torch.testing.assert_close(actual[j],expected)
        assert noise.step_count==step+1
    with pytest.raises(AssertionError): noise.next()
