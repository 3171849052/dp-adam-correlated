import copy
import itertools
from dataclasses import replace
import numpy as np
import pytest
import torch
from exp15b.config import Trial, METHODS, FINAL_SEEDS, SEARCH_SEED, SMOKE_SEED
from exp15b.optimizer import Optimizer
from exp15b.clipping import ScaledGhostModule, StandardGhostModule, LogicalBatch, clipped_microbatch
from exp15b.noise import BandInvMFNoise
from exp15b.privacy import calibrate, epsilon_from_mu, fixed_epoch_sensitivity
from exp15b.support import matrices, verify_assets


@pytest.mark.parametrize('method', METHODS)
def test_hand_reference_and_resume(method):
    p=torch.nn.Parameter(torch.tensor([1.,-2.],dtype=torch.float64))
    opt=Optimizer([p],method,lr=.03,betas=(.6,.8),eps_scale=.2)
    expected=p.detach().clone();u=torch.zeros(2,dtype=torch.float64);v=u.clone()
    resumed=None
    for t in range(1,5):
        # Noiseless query is deliberately different from its released value.
        q=torch.tensor([.4*t,-.3],dtype=torch.float64)
        z=torch.tensor([-.1,.2*t],dtype=torch.float64);released=q+z
        d=torch.sqrt(v/(1-.8**(t-1)))+.2 if t>1 else torch.full_like(v,.2)
        opt.freeze_scale()
        frozen=p._logical_denominator.clone()
        if method.startswith('stpex'):torch.testing.assert_close(frozen,d,rtol=1e-12,atol=1e-12)
        u=.6*u+.4*released
        expected-=.03*(u if method in ('momentum','momentum+iid') else u/(1-.6**t))
        if method=='stpex':v=.8*v+.2*(d*released)**2
        if method=='stpex2':v=.8*v+.2*released**2
        if method=='stpex-adam3':v=.8*v+.2*u**2
        p.grad=released.clone();opt.step()
        torch.testing.assert_close(p,expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(p._logical_denominator,frozen,rtol=0,atol=0)
        torch.testing.assert_close(opt.state[p]['exp_avg'],u,rtol=1e-12,atol=1e-12)
        if method.startswith('stpex'):torch.testing.assert_close(opt.state[p]['exp_avg_sq'],v,rtol=1e-12,atol=1e-12)
        if resumed:
            r,other=resumed;other.freeze_scale();r.grad=released.clone();other.step()
            torch.testing.assert_close(r,p,rtol=0,atol=0)
        if t==2:
            r=torch.nn.Parameter(p.detach().clone());other=Optimizer([r],method,.03,betas=(.6,.8),eps_scale=.2)
            other.load_state_dict(copy.deepcopy(opt.state_dict()));resumed=r,other


def test_first_step_bias_and_noisy_input_distinguish_variants():
    results={}
    for method in METHODS:
        p=torch.nn.Parameter(torch.zeros(1));o=Optimizer([p],method,.1,eps_scale=.5)
        o.freeze_scale();p.grad=torch.tensor([2.]);o.step();results[method]=(p.item(),o.state[p])
    assert results['momentum'][0]==pytest.approx(-.02)
    for method in METHODS[2:]:assert results[method][0]==pytest.approx(-.2)
    assert results['stpex'][1]['exp_avg_sq'].item()==pytest.approx(.001)
    assert results['stpex2'][1]['exp_avg_sq'].item()==pytest.approx(.004)
    assert results['stpex-adam3'][1]['exp_avg_sq'].item()==pytest.approx(.00004)


class FixedNoise:
    def __init__(self,params):self.params=list(params);self.step_count=0
    def next(self):
        self.step_count+=1
        return [torch.full_like(p,.13*self.step_count) for p in self.params]


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embed=torch.nn.Embedding(7,3,padding_idx=0);self.head=torch.nn.Linear(3,2)
    def forward(self,ids):return self.head(self.embed(ids).sum(1))


@pytest.mark.parametrize('method',METHODS)
def test_explicit_clipping_accumulation_previous_state(method):
    torch.manual_seed(42);raw=Tiny();reference=copy.deepcopy(raw)
    wrapper=ScaledGhostModule if method.startswith('stpex') else StandardGhostModule
    model=wrapper(raw,.7);opt=Optimizer(model.parameters(),method,.02,eps_scale=.4)
    other=Optimizer(reference.parameters(),method,.02,eps_scale=.4)
    noise=FixedNoise(model.parameters());logical=LogicalBatch(model,opt,2,4,noise,scaled=method.startswith('stpex'))
    ids=torch.tensor([[1,1,0],[2,3,2],[4,0,4],[1,2,3]]);targets=torch.tensor([0,1,1,0])
    for step in range(1,4):
        other.freeze_scale();sums=[torch.zeros_like(p) for p in reference.parameters()];norms=[]
        for x,y in zip(ids,targets):
            loss=torch.nn.functional.cross_entropy(reference(x[None]),y[None])
            gradients=torch.autograd.grad(loss,tuple(reference.parameters()))
            norm=torch.cat([(g*p._logical_scale).flatten() for g,p in zip(gradients,reference.parameters())]).norm()
            norms.append(norm);coeff=min(1.,.7/float(norm))
            for total,g in zip(sums,gradients):total.add_(g,alpha=coeff)
        for j in range(2):
            logical.begin_microbatch();clipped_microbatch(model,ids[j*2:j*2+2],targets[j*2:j*2+2],.7)
            torch.testing.assert_close(model.last_norms,torch.stack(norms[j*2:j*2+2]),rtol=2e-5,atol=2e-6)
            assert logical.finish_microbatch()==(j==1)
        for p,total in zip(reference.parameters(),sums):p.grad=(total*p._logical_scale+.13*step)/4
        other.step()
        assert noise.step_count==logical.optimizer_steps==step
        for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=3e-5,atol=2e-6)
    model.remove_hooks()


def test_nonfinite_does_not_partially_update():
    p=torch.nn.Parameter(torch.ones(2));q=torch.nn.Parameter(torch.ones(2))
    opt=Optimizer([p,q],'stpex2',.01);opt.freeze_scale();p.grad=torch.ones(2);q.grad=torch.tensor([float('nan'),0.])
    with pytest.raises(FloatingPointError):opt.step()
    torch.testing.assert_close(p,torch.ones(2));assert opt.state[p]['step']==0


def test_sensitivity_with_changing_directions_and_prefix():
    S=np.tril(np.array([[1.,0,0,0],[.2,1,0,0],[.1,.2,1,0],[.05,.1,.2,1]]))
    bound=fixed_epoch_sensitivity(S,2,2)
    for j in range(2):
        for signs in itertools.product((-1.,1.),repeat=2):
            x=np.zeros(4);x[[j,j+2]]=signs
            assert np.linalg.norm(S@x)<=bound+1e-12
    assert fixed_epoch_sensitivity(S,2,2,steps=2)<bound


@pytest.mark.parametrize('task,formal',[('cv',False),('cv',True),('nlp',False)])
def test_historical_matrix_calibration_noise_and_workload(task,formal):
    c=Trial(task,'stpex2',.01,smoke_phase='final' if formal else 'search')
    d,S,W,A,_=matrices(c)
    D=np.zeros_like(S)
    for lag,coef in enumerate(d):D+=np.diag(np.full(len(S)-lag,coef),-lag)
    np.testing.assert_allclose(S@D,np.eye(len(S)),atol=1e-12)
    rng=np.random.default_rng(3);g=rng.normal(size=len(S));u=0.;trajectory=[];x=0.
    for t,value in enumerate(g,1):u=.9*u+.1*value;x+=u/(1-.9**t);trajectory.append(x)
    np.testing.assert_allclose(A@g,trajectory,rtol=1e-12,atol=1e-12)
    uncorrected=matrices(replace(c,method='momentum'))[3]
    assert not np.allclose(uncorrected,A)
    for clip in (.1,1.,10.):
        trial=replace(c,clip=clip);cal=calibrate(S,trial.protocol())
        assert epsilon_from_mu(clip*cal['sensitivity']/cal['innovation_std_sum'],1e-5)==pytest.approx(8,abs=1e-7)
    # 30,000 parallel independent paths, FIR covariance and deterministic RNG.
    p=torch.zeros(30000);a=BandInvMFNoise([p],d,1.,6,7);b=BandInvMFNoise([p],d,1.,6,7)
    draws=[]
    for _ in range(6):
        x=a.next()[0];torch.testing.assert_close(x,b.next()[0],rtol=0,atol=0);draws.append(x.numpy())
    empirical=np.cov(draws);np.testing.assert_allclose(empirical,(D@D.T)[:6,:6],atol=.04,rtol=.05)


def test_assets_seed_stages_and_official_isolation():
    verify_assets()
    assert len(FINAL_SEEDS)==10 and set(FINAL_SEEDS).isdisjoint((SEARCH_SEED,SMOKE_SEED))
    with pytest.raises(ValueError):Trial('cv','stpex',.01,stage='search')
    import inspect
    from exp15b.data import nlp_datasets,cv_datasets
    assert "if cfg.stage in ('final','sweep')" in inspect.getsource(nlp_datasets)
    assert "if cfg.stage in ('final','sweep')" in inspect.getsource(cv_datasets)


def test_all_formal_orders_match_exp9():
    import json
    from exp15b import ROOT
    for task in ('cv','nlp'):
        with np.load(ROOT/f'exp9/results/{task}_split.npz') as split:
            indices=np.arange(50000) if task=='cv' else split['train'].copy()
        references={}
        for p in (ROOT/f'exp9/results/final/{task}').glob('*/summary.json'):
            r=json.loads(p.read_text())
            if r['method']=='dp-adam-momentum-bias-bandinvmf' and r['status']=='completed':references[r['seed']]=p.parent
        for seed in FINAL_SEEDS:
            expected=indices[torch.randperm(len(indices),generator=torch.Generator().manual_seed(seed)).numpy()]
            np.testing.assert_array_equal(expected,np.load(references[seed]/'train_order.npy'))


def test_statistics_and_selection_ties():
    from exp15b.report import stats
    from exp15b.search import rank
    s=stats([1.,2.,3.]);assert s['mean']==2. and s['sample_std']==1. and s['ci95_low']<2.<s['ci95_high']
    common=dict(status='completed',accuracy=.5,validation_loss=.2)
    rows=[dict(common,trial_id='b'),dict(common,trial_id='a'),dict(common,validation_loss=.3,trial_id='c')]
    assert [r['trial_id'] for r in rank(rows)]==['a','b','c']
