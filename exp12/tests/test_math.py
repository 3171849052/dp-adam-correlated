import copy
import json
import numpy as np
import pytest
import torch
from exp12 import ROOT, RESULTS
from exp12.config import Trial, METHOD
from exp12.bandinvmf import momentum_bias_workload, original_workload, bias_matrices, materialize, workload_error
from exp12.privacy import calibrate, epsilon_from_mu, fixed_epoch_sensitivity, gdp_delta
from exp12.noise import BandInvMFNoise
from exp12.clipping import StandardGhostModule, clipped_microbatch, LogicalBatch
from exp12.cv_model import ViTTiny, pretrained_vit, checkpoint_path
from exp12.rng import seed_all


def test_workload_definition_and_beta2_limit():
    n=13
    actual=momentum_bias_workload(n,.9)
    expected=np.zeros((n,n))
    for t in range(1,n+1):
        for j in range(1,t+1):
            expected[t-1,j-1]=sum(.9**(s-j)*np.sqrt(1-.999**s)/(1-.9**s) for s in range(j,t+1))
    np.testing.assert_allclose(actual,expected,rtol=2e-14)
    np.testing.assert_array_equal(actual,np.tril(actual))
    np.testing.assert_allclose(momentum_bias_workload(n,.9,0.),original_workload(n,.9),rtol=2e-14)
    assert actual[0,0]==pytest.approx(np.sqrt(.001)/.1)
    assert actual[0,0]!=actual[1,1]


@pytest.mark.parametrize('n',[225,250])
def test_full_matrix_objective_inverse_and_gdp(n):
    d,S,W,meta=bias_matrices(n)
    assert len(d)==4 and d[0]==1 and meta['converged']
    np.testing.assert_allclose(materialize(d,n)@S,np.eye(n),atol=3e-12)
    np.testing.assert_allclose(W,momentum_bias_workload(n,.9),rtol=1e-14)
    objective=fixed_epoch_sensitivity(S,5,n//5)**2*workload_error(W,d)
    assert objective==pytest.approx(meta['optimized_objective'],rel=1e-11)
    assert objective<=meta['initial_objective']*(1+1e-8)
    assert meta['new_filter']['new_objective']<=meta['old_filter']['new_objective']*(1+1e-8)
    # Shifted full-W quadratic agrees with direct multiplication for nonoptimal taps too.
    shifted=np.stack([np.pad(W[:,lag:],((0,0),(0,lag))) for lag in range(4)])
    Q=np.einsum('aij,bij->ab',shifted,shifted)/n
    arbitrary=np.array([1.,-.25,.13,-.08])
    assert arbitrary@Q@arbitrary==pytest.approx(workload_error(W,arbitrary),rel=1e-13)
    cfg=Trial('cv',METHOD,.001,30,stage='smoke',smoke_phase='final' if n==250 else 'search')
    for clip in (3,10,30,100,200):
        protocol=cfg.protocol();protocol['privacy']['max_grad_norm']=clip
        p=calibrate(S,protocol)
        mu=clip*p['sensitivity']/p['innovation_std_sum']
        assert gdp_delta(mu,8)==pytest.approx(1e-5,rel=1e-9)
        assert epsilon_from_mu(mu,1e-5)==pytest.approx(8,abs=1e-8)
        previous=0.
        for step in range(n//5,n+1,n//5):
            prefix=clip*fixed_epoch_sensitivity(S,5,n//5,steps=step)/p['innovation_std_sum']
            eps=epsilon_from_mu(prefix,1e-5)
            assert previous<=eps<=8+1e-8
            previous=eps
    params=[torch.zeros(3),torch.zeros(2,2)]
    noise=BandInvMFNoise(params,d,2.,n,46)
    gen=torch.Generator().manual_seed(46);history=[]
    for step in range(n):
        history.insert(0,[torch.randn(p.shape,generator=gen)*2 for p in params])
        expected=[sum(float(d[lag])*history[lag][j] for lag in range(min(4,step+1))) for j in range(2)]
        torch.manual_seed(step);torch.randn(7)
        for actual,want in zip(noise.next(),expected): torch.testing.assert_close(actual,want)
    assert noise.step_count==n
    with pytest.raises(AssertionError): noise.next()


def test_exp9_grid_ids_and_split_match():
    from exp12.run import grid_jobs
    jobs=grid_jobs()
    historical=json.loads((ROOT/'exp9/results/cv_grid.json').read_text())
    ids={r['trial_id'] for r in historical if r['method']==METHOD}
    assert len(jobs)==20 and {j.id for j in jobs}==ids
    with np.load(RESULTS/'cv_split.npz') as current, np.load(ROOT/'exp9/results/cv_split.npz') as old:
        for key in current.files: np.testing.assert_array_equal(current[key],old[key])


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_exact_cv_clipping_and_logical_adam(device):
    seed_all(42)
    base=ViTTiny(patch_size=4,image_size=16,embed_dim=12,depth=1,heads=3,mlp_ratio=2,num_classes=3).to(device)
    model=StandardGhostModule(copy.deepcopy(base),1.3)
    x=torch.randn(3,3,16,16,device=device);y=torch.tensor([0,1,2],device=device)
    totals=[torch.zeros_like(p) for p in base.parameters()];norms=[]
    for xi,yi in zip(x,y):
        base.zero_grad();torch.nn.functional.cross_entropy(base(xi[None]),yi[None]).backward()
        norm=sum(p.grad.square().sum() for p in base.parameters()).sqrt();norms.append(norm)
        for total,p in zip(totals,base.parameters()): total.add_(p.grad*min(1.,1.3/norm.item()))
    opt=torch.optim.Adam(model.parameters(),lr=.003,betas=(.9,.999),eps=1e-8)
    reference_opt=torch.optim.Adam(base.parameters(),lr=.003,betas=(.9,.999),eps=1e-8)
    noise=BandInvMFNoise(model.parameters(),[1.,-.2],.1,2,47)
    reference_noise=BandInvMFNoise(base.parameters(),[1.,-.2],.1,2,47)
    logical=LogicalBatch(model,opt,4,12,noise)
    for micro in range(4):
        logical.begin_microbatch();clipped_microbatch(model,x,y,1.3)
        torch.testing.assert_close(model.last_norms,torch.stack(norms),rtol=5e-5,atol=3e-6)
        for p,total in zip(model.parameters(),totals): torch.testing.assert_close(p.grad,total,rtol=5e-5,atol=3e-6)
        assert logical.finish_microbatch()==(micro==3)
        assert noise.step_count==(1 if micro==3 else 0)
    for p,total,z in zip(base.parameters(),totals,reference_noise.next()): p.grad=(4*total+z)/12
    reference_opt.step()
    for p,q in zip(model.parameters(),base.parameters()): torch.testing.assert_close(p,q,rtol=5e-5,atol=3e-6)
    assert logical.optimizer_steps==1 and logical.micro_steps==0
    model.remove_hooks()


def test_pretrained_vit_mapping():
    import timm
    from safetensors.torch import load_file
    seed_all(45);model,_=pretrained_vit();model.eval()
    ref=timm.create_model('vit_tiny_patch16_224',pretrained=False,num_classes=100).eval()
    weights={k:v for k,v in load_file(str(checkpoint_path())).items() if not k.startswith('head.')}
    weights.update({'head.weight':model.head.weight,'head.bias':model.head.bias})
    ref.load_state_dict(weights,strict=True)
    x=torch.randn(2,3,224,224)
    with torch.no_grad():
        torch.testing.assert_close(model(x),ref(x),rtol=3e-5,atol=1e-5)
        torch.testing.assert_close(model.double()(x.double()),ref.double()(x.double()),rtol=1e-9,atol=1e-10)


def test_exp9_selection_ties():
    from exp12.run import rank, select
    configs=sorted([Trial('cv',METHOD,lr,30) for lr in (.001,.003)],key=lambda j:j.id)
    rows=[dict(j.values(),trial_id=j.id,status='completed',accuracy=.5) for j in configs]
    assert rank(list(reversed(rows)))[0]['trial_id']==configs[0].id
    rechecks=[]
    from dataclasses import replace
    for cfg in configs:
        for seed in (20261102,20261103):
            j=replace(cfg,stage='recheck',seed=seed)
            rechecks.append(dict(j.values(),trial_id=j.id,status='completed',accuracy=.5))
    winner,_=select(configs,rows,rechecks)
    assert winner['base_id']==configs[0].id
