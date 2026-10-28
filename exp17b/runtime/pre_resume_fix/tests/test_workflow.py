import pytest
import torch
from exp17b.common import PROTOCOL, R, LR
from exp17b.pipeline import configurations, select_alpha, select_extra
from exp17b.privacy import calibrate
from opacus.accountants import RDPAccountant


def result(method,alpha,r,seed,loss):
    return dict(config=dict(method=method,alpha=alpha,r=r,seed=seed),validation_loss=loss,accuracy=.7)


def test_selection_and_phase_counts():
    rows=[result(m,a,R,s,1+a) for m in ['lc','gh'] for a in [.25,.5,1] for s in PROTOCOL['B']['seeds']]
    assert select_alpha(rows,'lc')==.25
    main=[('smooth',0.,R),('lc',.25,R),('gh',.5,R)]
    assert len(configurations('C',main,PROTOCOL['C']['seeds'],225))==15
    assert len(configurations('D',main,PROTOCOL['D']['seeds'],250))==30
    c=[result(m,0,R,s,1. if m=='smooth' else .95) for m in ['smooth','gh'] for s in PROTOCOL['C']['seeds']]
    c2=[result(m,0,r,s,1. if m=='smooth' else .9+r*.08) for r in [0.,1.] for m in ['smooth','gh'] for s in PROTOCOL['C2']['seeds']]
    assert select_extra(c,c2)['selected_r']==0
    for row in c2:
        if row['config']['method']=='gh':row['validation_loss']=1.1
    assert select_extra(c,c2)['selected_r'] is None


def test_short_budget_is_not_eight():
    c=calibrate('search');accountant=RDPAccountant()
    for _ in range(50):accountant.step(noise_multiplier=c['noise_multiplier'],sample_rate=c['sample_rate'])
    assert 0<accountant.get_epsilon(1e-5)<c['actual_epsilon']


def test_sampling_and_training_rng_replay():
    import random
    import numpy as np
    from exp17b.mechanism import poisson_indices
    random.seed(1);np.random.seed(1);torch.manual_seed(1);sampler=torch.Generator().manual_seed(5)
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),sampler.get_state())
    def draw():return random.random(),np.random.rand(),torch.rand(8),poisson_indices(45000,1000/45000,sampler)
    a=draw();random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);sampler.set_state(state[3]);b=draw()
    assert a[:2]==b[:2]
    assert torch.equal(a[2],b[2]) and torch.equal(a[3],b[3])
