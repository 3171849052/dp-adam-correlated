import copy
import numpy as np
import pytest
import torch
from exp3.geometry import Identity, NormScale, SpectralScale, snapshot_geometry, snapshot_inverse
from exp3.optimizer import muon_map
from exp3.diagnostics import (shape_factor, gain_metrics, probe_gain, temporal_probes,
                              nesterov_tangent, frozen_trajectory_diagnostic)


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.mark.parametrize('shape,rank',[((576,192),192),((192,576),192),((192,192),192),
                                       ((576,192),17),((192,576),17),((192,192),17)])
def test_rectangular_active_spectrum_and_exact_identity(shape,rank):
    torch.manual_seed(7)
    h=torch.randn(shape[0],rank,dtype=torch.double)@torch.randn(rank,shape[1],dtype=torch.double)
    s=SpectralScale(h,4,.1)
    g=torch.randn(shape,dtype=torch.double)
    torch.testing.assert_close(s.inverse(s.transform(g)),g,rtol=1e-10,atol=1e-10)
    assert s.pairwise_min>=.25-1e-12 and s.pairwise_max<=4+1e-12
    assert len(s.active_gain)==rank
    assert torch.quantile(2*s.active_gain.log(),.5).item()==pytest.approx(0,abs=1e-11)
    one=SpectralScale(h,1,.1)
    assert torch.equal(one.transform(g),g) and torch.equal(one.inverse(g),g)


def test_structural_null_dimensions_do_not_set_center():
    active=torch.diag(torch.tensor([5.,3.,2.,1.],dtype=torch.double))
    base=SpectralScale(active,16,.1)
    for shape in ((12,4),(4,12),(12,12)):
        padded=torch.zeros(shape,dtype=torch.double)
        padded[:4,:4]=active
        other=SpectralScale(padded,16,.1)
        assert other.delta==pytest.approx(base.delta)
        assert other.normalization_center==pytest.approx(base.normalization_center)
        torch.testing.assert_close(other.active_gain,base.active_gain,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(other.left[:4,:4],base.left,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(other.right[:4,:4],base.right,rtol=1e-12,atol=1e-12)
    padded=torch.zeros(12,4,dtype=torch.double);padded[:4]=active
    other=SpectralScale(padded,16,.1)
    assert abs(torch.quantile(other.left_gain.log(),.5).item())>.05


@pytest.mark.parametrize('shape',[(6,2),(2,6),(3,3)])
def test_update_gain_shape_factor_and_noise_formula(shape):
    torch.manual_seed(8)
    h,e=torch.randn(shape,dtype=torch.double),torch.randn(shape,dtype=torch.double)
    s=NormScale(h,.4)
    phi=probe_gain(h,e,s)
    row=gain_metrics([phi,phi*2],shape,123.,1000)
    tangent=s.inverse(e)
    fd=shape_factor(shape)*(muon_map(h+1e-6*tangent)-muon_map(h-1e-6*tangent))/(2e-6)
    assert shape_factor(shape)*phi==pytest.approx(float(fd.norm()/e.norm()),rel=3e-6)
    assert row['update_gain_mean']==pytest.approx(shape_factor(shape)*phi*1.5)
    assert row['noise_weighted_update_gain_mean']==pytest.approx(.123*row['update_gain_mean'])
    assert row['noise_weighted_update_gain_cv']==pytest.approx(row['update_gain_cv'])
    private2=gain_metrics([phi,phi*2],shape,246.,1000)
    assert private2['noise_weighted_update_gain_std']==pytest.approx(2*row['noise_weighted_update_gain_std'])
    nonprivate=gain_metrics([phi],shape,None,1000)
    assert all(v is None for k,v in nonprivate.items() if k.startswith('noise_weighted'))


def test_actual_nesterov_tangent_impulse_and_finite_difference():
    e=torch.tensor([[1.,2.]])
    momentum,dh=nesterov_tangent(e,torch.zeros_like(e))
    torch.testing.assert_close(dh,.0975*e)
    momentum,dh=nesterov_tangent(torch.zeros_like(e),momentum)
    torch.testing.assert_close(dh,.045125*e)
    torch.manual_seed(9)
    grads=[torch.randn(3,2,dtype=torch.double) for _ in range(4)]
    noises=[torch.randn_like(g) for g in grads]
    m0=torch.zeros_like(grads[0]); m1=m0.clone();dm=m0.clone()
    for g,e in zip(grads,noises):
        m0=torch.lerp(m0,g,.05);m1=torch.lerp(m1,g+1e-6*e,.05)
        h0=torch.lerp(g,m0,.95);h1=torch.lerp(g+1e-6*e,m1,.95)
        dm,dh=nesterov_tangent(e,dm)
        torch.testing.assert_close((h1-h0)/1e-6,dh,rtol=1e-8,atol=1e-8)


def trajectory(kind):
    torch.manual_seed(10)
    frames=[]
    for step in range(4):
        h=torch.randn(4,2)
        s={'standard':Identity(),'normscale':NormScale(h,.5),'spectralscale':SpectralScale(h,3,.1)}[kind]
        frames.append(dict(step=step+1,h=h,geometry=snapshot_geometry(s)))
        e=torch.randn_like(h)
        torch.testing.assert_close(snapshot_inverse(frames[-1]['geometry'],e),s.inverse(e))
    return {'test_layer':frames}


def test_frozen_determinism_same_temporal_probes_and_noise_scaling():
    d=np.tril(np.ones((4,4)))*.2+np.eye(4)*.8
    reports={}
    rng_before=torch.get_rng_state().clone()
    for kind in ('standard','normscale','spectralscale'):
        t=trajectory(kind)
        before=torch.get_rng_state().clone()
        a=frozen_trajectory_diagnostic(t,d,2.,1000,.01,'cpu',probes=2)
        b=frozen_trajectory_diagnostic(t,d,2.,1000,.01,'cpu',probes=2)
        assert a==b and torch.equal(before,torch.get_rng_state())
        assert a['actual_dynamics']=='EMA Nesterov beta=.95'
        assert a['mf_design_workload']=='ordinary momentum beta=.9'
        reports[kind]=a
        doubled=frozen_trajectory_diagnostic(t,d,4.,1000,.01,'cpu',probes=2)
        assert doubled['aggregate']['final_cumulative_rmse']['mean']==pytest.approx(2*a['aggregate']['final_cumulative_rmse']['mean'])
    assert len({r['temporal_probe_sha256']['test_layer'] for r in reports.values()})==1


def test_frozen_prefix_rmse_matches_explicit_first_order_recurrence():
    t=trajectory('normscale');frames=t['test_layer'];shape=(4,2)
    d=np.eye(4);sigma=2.;batch=1000;lr=.01
    z=temporal_probes('test_layer',shape,4,1)[0]
    dm=torch.zeros(shape);cumulative=torch.zeros(shape);prefix=[]
    for frame,e in zip(frames,z*sigma/batch):
        e=snapshot_inverse(frame['geometry'],e)
        dm,dh=nesterov_tangent(e,dm)
        h=frame['h'].double();dh=dh.double()
        derivative=(muon_map(h+1e-5*dh)-muon_map(h-1e-5*dh))/(2e-5)
        cumulative-=lr*shape_factor(shape)*derivative.float()
        prefix.append(float(cumulative.square().mean().sqrt()))
    report=frozen_trajectory_diagnostic(t,d,sigma,batch,lr,'cpu',probes=1)
    assert report['aggregate']['final_cumulative_rmse']['mean']==pytest.approx(prefix[-1],rel=2e-5)
    assert report['aggregate']['mean_prefix_rmse']['mean']==pytest.approx(np.mean(prefix),rel=2e-5)
