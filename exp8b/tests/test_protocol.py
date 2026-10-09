from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from exp8b import RESULTS
from exp8b.config import *
from exp8b.bandinvmf import build_matrices,bias_matrices,momentum_bias_workload,materialize,workload_error
from exp8b.privacy import calibrate,fixed_epoch_sensitivity,epsilon_from_mu
from exp8b.data import datasets,split_indices,official_validation


def test_locked_protocol():
    for kw in (dict(physical_batch_size=500),dict(max_length=64),dict(total_steps=250),dict(adam_eps=.1),dict(epochs=1)):
        with pytest.raises(AssertionError): Config(**kw)
    with pytest.raises(AssertionError): Config(method=BIAS_SCALE)
    with pytest.raises(AssertionError): Config(eps_scale=.1)
    assert len(METHODS)==7


@pytest.mark.parametrize('method',METHODS)
def test_full_strategy_accounting(method):
    d,S,W=build_matrices(METHODS[method]['noise'],310,4,.9)
    np.testing.assert_allclose(materialize(d,310)@S,np.eye(310),atol=2e-12)
    assert S.shape==W.shape==(310,310) and len(d)==(1 if method==IID else 4)
    for C in (1.,30.,1000.):
        cfg=Config(method=method,C=C,eps_scale=.03 if method.endswith('-scale') else None)
        p=calibrate(S,cfg.protocol())
        assert epsilon_from_mu(C*p['sensitivity']/p['innovation_std_sum'],1e-5)==pytest.approx(8,abs=1e-8)
    if method.endswith('-scale'):
        original=build_matrices(METHODS[method[:-6]]['noise'],310,4,.9)
        for a,b in zip((d,S,W),original): np.testing.assert_array_equal(a,b)
    if method==IID: assert fixed_epoch_sensitivity(S,5,62)==pytest.approx(np.sqrt(5))


def test_non_toeplitz_bias_full_objective():
    W=momentum_bias_workload(8,.9)
    expected=np.zeros((8,8))
    for t in range(1,9):
        for j in range(1,t+1): expected[t-1,j-1]=sum(.9**(u-j)/(1-.9**u) for u in range(j,t+1))
    np.testing.assert_allclose(W,expected,rtol=1e-14)
    d,S,W,meta=bias_matrices()
    direct=fixed_epoch_sensitivity(S,5,62)**2*workload_error(W,d)
    assert direct==pytest.approx(meta['optimized_objective'],rel=1e-11)
    assert meta['converged'] and direct<=meta['initial_objective']
    assert not np.allclose(W,materialize(W[:,0],310))


def test_split_and_fixed_participation():
    train,valid=datasets()
    assert len(train)==62000 and len(valid)==5349 and train.tensors[0].shape==(62000,128)
    with np.load(RESULTS/'split.npz') as s:
        order,holdout=s['train'],s['search_validation']
    np.testing.assert_array_equal(np.sort(np.r_[order,holdout]),np.arange(67349))
    for offset in (0,999,61999):
        np.testing.assert_array_equal(np.where(np.tile(order,5)==order[offset])[0]//1000,offset//1000+62*np.arange(5))
    manifest=json.loads((RESULTS/'data_manifest.json').read_text())
    assert manifest['split_sha256']==file_hash(RESULTS/'split.npz')
    assert manifest['tokens_sha256']==file_hash(RESULTS/'tokens_128.pt')
    assert not manifest['official_validation_used']
    ids,mask,segments,labels=train.tensors
    assert ((ids==0)==(mask==0)).all() and not segments.any() and set(labels.tolist())=={0,1}
    with pytest.raises(AssertionError): official_validation('search')
    with pytest.raises(AssertionError): official_validation('smoke')


def test_fir_retains_innovations():
    from exp8b.noise import BandInvMFNoise
    p=torch.nn.Parameter(torch.zeros(5));d=np.array([1.,-.3,.1,-.02])
    source=BandInvMFNoise([p],d,2.,8,77)
    generator=torch.Generator().manual_seed(77);history=[]
    for step in range(8):
        history.insert(0,torch.randn(5,generator=generator)*2)
        torch.testing.assert_close(source.next()[0],sum(float(a)*b for a,b in zip(d,history)),rtol=2e-6,atol=2e-6)
        assert source.step_count==step+1 and len(source.history)<=3
    with pytest.raises(AssertionError): source.next()


def test_pretrained_full_finetuning_and_paired_dropout():
    from exp8b.model import seed_all,make_model,digest
    seed_all(SEED);a=make_model();head=digest(a.classifier)
    seed_all(SEED);b=make_model();assert digest(a)==digest(b)
    state=torch.load(MODEL_PATH/'pytorch_model.bin',weights_only=True)
    for name,p in a.bert.named_parameters():
        torch.testing.assert_close(p,state['bert.'+name],rtol=0,atol=0);assert p.requires_grad
    assert a.bert.config.hidden_dropout_prob==.1
    seed_all(SEED+1);assert head!=digest(make_model().classifier)
    ids=torch.tensor([[101,2023,2003,102,0,0],[101,2204,102,0,0,0]])
    x=(ids,(ids!=0).long(),torch.zeros_like(ids))
    torch.manual_seed(SEED+100000);out_a=a(x)
    torch.manual_seed(SEED+100000);out_b=b(x)
    torch.testing.assert_close(out_a,out_b,rtol=0,atol=0)
