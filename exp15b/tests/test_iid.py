import copy
import itertools
from dataclasses import replace
import numpy as np
import pytest
import torch
from exp15b import ROOT, BASE
from exp15b.config import Trial, METHODS, SEARCH_SEED
from exp15b.optimizer import Optimizer
from exp15b.noise import BandInvMFNoise
from exp15b.privacy import calibrate, epsilon_from_mu, fixed_epoch_sensitivity
from exp15b.support import matrices
from exp15b.search import extension_trials, rank
from exp15b.audit import verify_equivalent_sources


@pytest.mark.parametrize('method',METHODS)
def test_exact_optimizer_equivalence_to_historical_source(method):
    # Execute just the old class definition: never import the historical initializer.
    namespace={}
    old=(ROOT/'exp15/optimizer.py').read_text().replace('from exp15.config import METHODS',
        "METHODS=('momentum','momentum-bias','stpex','stpex2','stpex-adam3')")
    exec(compile(old,'historical_optimizer','exec'),namespace)
    p=torch.nn.Parameter(torch.tensor([1.,-2.],dtype=torch.float64))
    q=torch.nn.Parameter(p.detach().clone())
    current=Optimizer([p],method,.37)
    previous=namespace['Optimizer']([q],'momentum' if method=='momentum+iid' else method,.37)
    generator=torch.Generator().manual_seed(71)
    for _ in range(10):
        current.freeze_scale();previous.freeze_scale()
        grad=torch.randn(p.shape,generator=generator,dtype=p.dtype)
        p.grad=grad.clone();q.grad=grad.clone();current.step();previous.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        for key,value in current.state[p].items():
            if isinstance(value,torch.Tensor):torch.testing.assert_close(value,previous.state[q][key],rtol=0,atol=0)
            else:assert value==previous.state[q][key]


@pytest.mark.parametrize('task,phase',[('cv','search'),('cv','final'),('nlp','search')])
def test_iid_full_schedule_privacy_and_arbitrary_participation(task,phase):
    c=Trial(task,'momentum+iid',.01,clip=3.,smoke_phase=phase)
    d,S,W,A,source=matrices(c)
    np.testing.assert_array_equal(S,np.eye(c.total_steps))
    np.testing.assert_array_equal(d,[1.])
    assert fixed_epoch_sensitivity(S,5,c.spacing)==pytest.approx(np.sqrt(5))
    # Identity supports any at-most-one-per-epoch schedule, even changing positions.
    rng=np.random.default_rng(7)
    for _ in range(25):
        indices=[e*c.spacing+int(rng.integers(c.spacing)) for e in range(5)]
        directions=rng.normal(size=(5,7));directions/=np.linalg.norm(directions,axis=1)[:,None]
        queries=np.zeros((c.total_steps,7));queries[indices]=directions
        assert np.linalg.norm(S@queries)==pytest.approx(np.sqrt(5))
    cal=calibrate(S,c.protocol())
    assert cal['innovation_std_sum']==pytest.approx(c.clip*np.sqrt(5)/cal['target_mu'])
    assert epsilon_from_mu(c.clip*np.sqrt(5)/cal['innovation_std_sum'],1e-5)==pytest.approx(8.,abs=1e-7)
    correlated=calibrate(matrices(replace(c,method='momentum'))[1],replace(c,method='momentum').protocol())
    assert cal['innovation_std_sum']!=pytest.approx(correlated['innovation_std_sum'])
    np.testing.assert_array_equal(A,matrices(replace(c,method='momentum'))[3])


def test_iid_independence_covariance_rng_and_reproduction():
    p=torch.zeros(90000)
    a=BandInvMFNoise([p],[1.],2.7,8,42)
    b=BandInvMFNoise([p],[1.],2.7,8,42)
    torch.manual_seed(123);global_state=torch.get_rng_state().clone()
    draws=[]
    for _ in range(8):
        x=a.next()[0];torch.testing.assert_close(x,b.next()[0],rtol=0,atol=0)
        assert a.history==[]
        draws.append(x.numpy())
    torch.testing.assert_close(global_state,torch.get_rng_state(),rtol=0,atol=0)
    covariance=np.cov(draws)
    np.testing.assert_allclose(covariance,2.7**2*np.eye(8),atol=.09,rtol=.015)
    assert np.max(np.abs(np.mean(draws,axis=1)))<.035


def test_predeclared_extension_and_fixed_clip_subset():
    jobs=extension_trials()
    assert len(jobs)==150 and len({c.id for c in jobs})==150
    assert sum(c.task=='cv' for c in jobs)==60
    for task in ('cv','nlp'):
        grids={m:{(c.lr,c.clip) for c in jobs if c.task==task and c.method==m} for m in METHODS}
        assert grids['momentum']==grids['momentum+iid']
    assert all(c.seed==20261101 and c.stage=='search' for c in jobs)
    for seed in (20261102,20261103):
        review=replace(jobs[0],stage='review',seed=seed)
        assert not review.formal and review.total_steps==225
    with pytest.raises(ValueError):replace(jobs[0],stage='review',seed=20261111)
    verify_equivalent_sources()


def test_gpu_three_is_excluded():
    import inspect
    from exp15b import scheduler, train
    assert 'for gpu in (0,1,2):' in inspect.getsource(scheduler.run_batch)
    assert "not in ('0','1','2')" in inspect.getsource(train.run)
