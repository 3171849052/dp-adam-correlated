import copy
import numpy as np
import pytest
import torch
from exp9.config import Trial, METHODS
from exp9.bandinvmf import build_matrices, materialize, momentum_bias_workload, bias_matrices, workload_error
from exp9.privacy import calibrate, epsilon_from_mu, fixed_epoch_sensitivity
from exp9.noise import BandInvMFNoise
from exp9.clipping import StandardGhostModule, ScaledGhostModule, freeze_adam_scale, clipped_microbatch
from exp9.cv_model import ViTTiny, pretrained_vit, checkpoint_path
from exp9.nlp_model import seed_all

@pytest.mark.parametrize('horizon',[225,250,310])
@pytest.mark.parametrize('method',METHODS)
def test_complete_horizon_matrices_noise_and_gdp(horizon,method):
    cfg=Trial('nlp' if horizon==310 else 'cv',method,.001,30,.1 if method.endswith('-scale') else None,
              stage='smoke',smoke_phase='final' if horizon==250 else 'search')
    assert cfg.total_steps==horizon and cfg.spacing==horizon//5
    d,S,W=build_matrices(cfg.noise,horizon,4,.9)
    np.testing.assert_allclose(materialize(d,horizon)@S,np.eye(horizon),atol=3e-12)
    assert np.isfinite(S).all() and W.shape==(horizon,horizon)
    if method==METHODS[0]: np.testing.assert_array_equal(W,np.eye(horizon))
    else: assert len(d)==4
    if cfg.scaled:
        for actual,expected in zip((d,S,W),build_matrices(cfg.noise,horizon,4,.9)): np.testing.assert_array_equal(actual,expected)
    for epsilon in (2,4,8,16):
        for C in (.1,100,300):
            protocol=cfg.protocol();protocol['privacy'].update(epsilon=epsilon,max_grad_norm=C)
            p=calibrate(S,protocol)
            assert epsilon_from_mu(C*p['sensitivity']/p['innovation_std_sum'],1e-5)==pytest.approx(epsilon,abs=1e-8)
            partial=C*fixed_epoch_sensitivity(S,5,horizon//5,steps=2)/p['innovation_std_sum']
            assert epsilon_from_mu(partial,1e-5)<=epsilon
    # Full trajectories: one independent innovation per parameter per logical step.
    params=[torch.zeros(3),torch.zeros(2,2)];noise=BandInvMFNoise(params,d,2.,horizon,46)
    gen=torch.Generator().manual_seed(46);history=[]
    for step in range(horizon):
        innovation=[torch.randn(p.shape,generator=gen)*2 for p in params];history.insert(0,innovation)
        expected=[sum(float(d[lag])*history[lag][j] for lag in range(min(len(d),step+1))) for j in range(2)]
        for actual,want in zip(noise.next(),expected): torch.testing.assert_close(actual,want)
        assert noise.step_count==step+1
    with pytest.raises(AssertionError): noise.next()

def test_workloads_against_definitions():
    n=8;W=momentum_bias_workload(n,.9)
    expected=np.zeros((n,n))
    for t in range(1,n+1):
        for j in range(1,t+1): expected[t-1,j-1]=sum(.9**(u-j)/(1-.9**u) for u in range(j,t+1))
    np.testing.assert_allclose(W,expected,rtol=1e-14)
    assert W[0,0]!=W[1,1]
    _,_,sgd=build_matrices('prefix_bandinvmf',225,4,.9)
    np.testing.assert_array_equal(sgd,np.tril(np.ones((225,225))))
    _,_,mom=build_matrices('momentum_bandinvmf',225,4,.9)
    for t in (0,1,5,224):
        for j in (0,t): assert mom[t,j]==pytest.approx(sum(.9**u for u in range(t-j+1)))

@pytest.mark.parametrize('horizon',[225,250,310])
def test_bias_optimization_full_non_toeplitz_workload(horizon):
    d,S,W,meta=bias_matrices(horizon,4,.9)
    direct=fixed_epoch_sensitivity(S,5,horizon//5)**2*workload_error(W,d)
    assert direct==pytest.approx(meta['optimized_objective'],rel=1e-11)
    assert meta['converged'] and meta['full_workload_shape']==[horizon,horizon]
    assert direct<=meta['initial_objective']*(1+1e-8)
    assert not np.isclose(workload_error(W,d),workload_error(materialize(W[:,0],horizon),d),rtol=.01)

@pytest.mark.parametrize('scaled',[False,True])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_cv_exact_clipping(scaled,device):
    seed_all(42)
    base=ViTTiny(patch_size=4,image_size=16,embed_dim=12,depth=1,heads=3,mlp_ratio=2,num_classes=3).to(device)
    C=1.3;model=(ScaledGhostModule if scaled else StandardGhostModule)(copy.deepcopy(base),C)
    opt=torch.optim.Adam(model.parameters())
    for p in model.parameters(): opt.state[p].update(step=torch.tensor(3.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.rand_like(p)*.001)
    if scaled: freeze_adam_scale(opt,.1)
    x=torch.randn(3,3,16,16,device=device);y=torch.tensor([0,1,2],device=device)
    totals=[torch.zeros_like(p) for p in base.parameters()];norms=[]
    for xi,yi in zip(x,y):
        base.zero_grad();torch.nn.functional.cross_entropy(base(xi[None]),yi[None]).backward()
        norm=sum((p.grad*q._logical_scale if scaled else p.grad).square().sum() for p,q in zip(base.parameters(),model.parameters())).sqrt()
        norms.append(norm)
        for total,p in zip(totals,base.parameters()): total.add_(p.grad*min(1.,C/norm.item()))
    clipped_microbatch(model,x,y,C)
    torch.testing.assert_close(model.last_norms,torch.stack(norms),rtol=5e-5,atol=3e-6)
    for p,total in zip(model.parameters(),totals): torch.testing.assert_close(p.grad,total,rtol=5e-5,atol=3e-6)
    model.remove_hooks()

def test_pretrained_vit_matches_timm_logits():
    import timm
    from safetensors.torch import load_file
    seed_all(45);model,_=pretrained_vit();model.eval()
    reference=timm.create_model('vit_tiny_patch16_224',pretrained=False,num_classes=100).eval()
    weights=load_file(str(checkpoint_path()))
    weights={k:v for k,v in weights.items() if not k.startswith('head.')}
    weights.update({'head.weight':model.head.weight,'head.bias':model.head.bias})
    reference.load_state_dict(weights,strict=True)
    images=torch.randn(2,3,224,224)
    with torch.no_grad():
        # Patch convolution vs unfolded Linear have different FP32 accumulation order.
        torch.testing.assert_close(model(images),reference(images),rtol=3e-5,atol=1e-5)
        torch.testing.assert_close(model.double()(images.double()),reference.double()(images.double()),rtol=1e-9,atol=1e-10)

def test_noise_independent_of_global_dropout_rng():
    p=[torch.zeros(5)];a=BandInvMFNoise(p,[1,-.2],2,3,47);b=BandInvMFNoise(p,[1,-.2],2,3,47)
    for _ in range(3):
        first=a.next();torch.manual_seed(100);torch.randn(1000)
        torch.testing.assert_close(first[0],b.next()[0])
