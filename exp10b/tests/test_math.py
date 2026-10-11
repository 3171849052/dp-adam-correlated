import hashlib
import types
import numpy as np
import pytest
import torch
from exp10b import ROOT,RESULTS
from exp10b.config import Trial,LAMBDAS,save_json
from exp10b.matrices import get_matrix,baseline_matrix,components,weights,optimize
from exp10b.bandinvmf import momentum_bias_workload
from exp10b.noise import materialize,BandInvMFNoise
from exp10b.privacy import calibrate,epsilon_from_mu,fixed_epoch_sensitivity,gdp_delta

@pytest.mark.parametrize('n',[225,250,310])
@pytest.mark.parametrize('lam',LAMBDAS)
def test_inverse_objectives_and_full_gdp(n,lam):
    d,S,W,meta=get_matrix(n,lam);D=materialize(d,n)
    np.testing.assert_allclose(D@S,np.eye(n),atol=3e-12)
    np.testing.assert_allclose(S@D,np.eye(n),atol=3e-12)
    assert len(d)==4 and d[0]==1.
    gram=S.T@S
    sens=max(np.abs(gram[np.ix_(idx,idx)]).sum() for j in range(n//5)
        for idx in [j+(n//5)*np.arange(5)])**.5
    assert sens==pytest.approx(meta['sensitivity'],rel=1e-12)
    assert meta['J1']==pytest.approx(sens**2*np.sum((W@D)**2)/n,rel=1e-12)
    w=weights(n);cov=meta['sigma_avg_C1']**2*(D@D.T)
    direct=2*np.mean([row@(cov*cov)@row for row in w])
    diag=2*np.mean([np.sum(row**2*np.diag(cov)**2) for row in w])
    cross=4*np.mean([sum(row[s]*row[u]*cov[s,u]**2 for s in range(n) for u in range(s+1,min(n,s+4))) for row in w])
    for key,value in [('J2',direct),('J2_diag',diag),('J2_cross',cross),('R_corr',direct/diag)]:
        assert meta[key]==pytest.approx(value,rel=1e-12)
    assert direct==pytest.approx(diag+cross,rel=1e-12)
    assert meta['normalized_objective']<=1+lam+1e-8
    assert meta['normalized_objective']==pytest.approx(meta['J1_ratio']+lam*meta['J2_ratio'],rel=1e-10)
    assert meta['C_clip_objective']==1.
    cfg=Trial('nlp' if n==310 else 'cv',lam,stage='smoke',smoke_phase='final' if n==250 else 'search')
    p=calibrate(S,cfg.protocol())
    assert epsilon_from_mu(cfg.C*sens/p['innovation_std_sum'],1e-5)==pytest.approx(8.,abs=1e-8)
    assert gdp_delta(p['target_mu'],8.)==pytest.approx(1e-5,rel=1e-9)
    prefix=[fixed_epoch_sensitivity(S,5,n//5,steps=t) for t in (1,n//5,2*n//5,n)]
    assert prefix==sorted(prefix)
    for C in (1.,30.):
        protocol=cfg.protocol();protocol['privacy']['max_grad_norm']=C
        calibration=calibrate(S,protocol)
        assert epsilon_from_mu(C*sens/calibration['innovation_std_sum'],1e-5)==pytest.approx(8.,abs=1e-8)
    scaled=components(d,n,W)
    assert scaled['J2']==pytest.approx(meta['J2'],rel=1e-12)

@pytest.mark.parametrize('n',[225,250,310])
def test_lambda_zero_is_exact_stored_exp9_never_optimized(n,monkeypatch):
    from exp10b import matrices
    original=baseline_matrix(n)
    monkeypatch.setattr(matrices,'minimize',lambda *a,**kw:pytest.fail('lambda zero must not optimize'))
    d,meta=optimize(n,0.)
    np.testing.assert_array_equal(d,original[0])
    actual=get_matrix(n,0.)
    for a,b in zip(actual[:3],original[:3]): np.testing.assert_array_equal(a,b)
    assert meta['iterations']==0 and actual[3]['J1_ratio']==actual[3]['J2_ratio']==1.

def test_workload_and_beta2_bias_correction():
    n=12;expected=np.zeros((n,n));w=np.zeros((n,n))
    for t in range(1,n+1):
        for s in range(1,t+1):
            expected[t-1,s-1]=sum(.9**(u-s)/(1-.9**u) for u in range(s,t+1))
            w[t-1,s-1]=(1-.999)*.999**(t-s)/(1-.999**t)
    np.testing.assert_allclose(momentum_bias_workload(n,.9),expected,rtol=1e-14)
    np.testing.assert_allclose(weights(n),w,rtol=2e-14)
    np.testing.assert_allclose(weights(n).sum(axis=1),1.,atol=2e-14)

@pytest.mark.parametrize('n',[225,250,310])
@pytest.mark.parametrize('lam',LAMBDAS)
def test_J2_monte_carlo_each_horizon_lambda(n,lam):
    d,S,W,meta=get_matrix(n,lam);D=materialize(d,n)
    Z=np.random.default_rng(813+n).standard_normal((n,12000))
    second=weights(n)@((meta['sigma_avg_C1']*(D@Z))**2)
    empirical=float(second.var(axis=1,ddof=1).mean())
    assert empirical==pytest.approx(meta['J2'],rel=.065)
    save_json(RESULTS/'math_validation'/f'T{n}_lambda_{lam:g}_mc.json',dict(
        horizon=n,module=lam,samples=12000,analytic_J2=meta['J2'],empirical_J2=empirical,
        relative_error=empirical/meta['J2']-1.,status='passed'))

@pytest.mark.parametrize('n',[225,250,310])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_common_Z_full_horizon_and_historical_noise(n,device):
    params=[torch.zeros(3,device=device),torch.zeros(2,2,device=device)]
    noises=[BandInvMFNoise(params,get_matrix(n,lam)[0],1.2+i,n,46) for i,lam in enumerate(LAMBDAS)]
    historical=types.ModuleType('historical_noise')
    exec(compile((ROOT/'exp9/noise.py').read_text(),str(ROOT/'exp9/noise.py'),'exec'),historical.__dict__)
    ref=historical.BandInvMFNoise(params,baseline_matrix(n)[0],1.2,n,46)
    gen=torch.Generator(device=device).manual_seed(46);history=[]
    for step in range(n):
        history.insert(0,[torch.randn(p.shape,device=device,generator=gen) for p in params]);history=history[:4]
        actuals=[noise.next() for noise in noises];old=ref.next()
        for a,b in zip(actuals[0],old): torch.testing.assert_close(a,b,rtol=0,atol=0)
        for noise,actual in zip(noises,actuals):
            for j in range(2):
                # Reproduce the historical FP32 order: scale innovations, then FIR.
                expected=history[0][j]*noise.innovation_std*float(noise.coefficients[0])
                for lag in range(1,min(4,step+1)):
                    expected.add_(history[lag][j]*noise.innovation_std,alpha=float(noise.coefficients[lag]))
                torch.testing.assert_close(actual[j],expected,rtol=0,atol=0)
        torch.manual_seed(123+step);torch.randn(7,device=device)
    assert len({noise.first_innovation_sha256 for noise in noises})==1
    assert all(noise.step_count==n for noise in noises)
    for noise in noises:
        with pytest.raises(AssertionError): noise.next()

def test_standard_adam_bias_corrections_and_diagnostics():
    from exp10b.optim import PreconditionerDiagnostics
    p=torch.tensor([.5,-.4],dtype=torch.float64,requires_grad=True);expected=p.detach().clone()
    opt=torch.optim.Adam([p],lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    m=torch.zeros_like(p);v=m.clone();diagnostics=PreconditionerDiagnostics()
    for step in range(1,12):
        gradient=torch.tensor([.1*step,-.2/step],dtype=p.dtype);p.grad=gradient
        m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
        expected-=.003*(m/(1-.9**step))/(torch.sqrt(v/(1-.999**step))+1e-8)
        opt.step();torch.testing.assert_close(p,expected,rtol=1e-12,atol=1e-12)
        state_before=p.detach().clone();row=diagnostics.record(opt)
        torch.testing.assert_close(p,state_before,rtol=0,atol=0)
        assert np.isfinite(list(row.values())).all()
