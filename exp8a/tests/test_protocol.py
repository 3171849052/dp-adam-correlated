from dataclasses import replace
import json
import numpy as np
import pytest
from exp8a import ROOT, RESULTS
from exp8a.config import Config, file_hash
from exp8a.bandinvmf import bias_matrices, momentum_bias_workload, materialize, workload_error
from exp8a.privacy import fixed_epoch_sensitivity, calibrate, epsilon_from_mu
from exp8a.data import datasets


def test_full_bias_workload_and_privacy():
    W=momentum_bias_workload(8,.9)
    expected=np.zeros((8,8))
    for t in range(1,9):
        for j in range(1,t+1): expected[t-1,j-1]=sum(.9**(k-j)/(1-.9**k) for k in range(j,t+1))
    np.testing.assert_allclose(W,expected,rtol=1e-14)
    d,S,W,meta=bias_matrices()
    assert len(d)==4 and W.shape==(310,310) and meta['converged']
    np.testing.assert_allclose(materialize(d,310)@S,np.eye(310),atol=2e-12)
    objective=fixed_epoch_sensitivity(S,5,62)**2*workload_error(W,d)
    assert objective==pytest.approx(meta['optimized_objective'],rel=1e-11)
    assert objective<=meta['initial_objective']
    assert not np.allclose(W,materialize(W[:,0],310))
    for C in (1.,3.,10.):
        cfg=Config(C=C)
        p=calibrate(S,cfg.protocol())
        assert epsilon_from_mu(C*p['sensitivity']/p['innovation_std_sum'],1e-5)==pytest.approx(8.,abs=1e-8)


def test_fixed_split_and_participation():
    with np.load(RESULTS/'split.npz') as s:
        train,valid=s['train'],s['search_validation']
    assert (len(train),len(valid))==(62000,5349)
    np.testing.assert_array_equal(np.sort(np.r_[train,valid]),np.arange(67349))
    all_epochs=np.tile(train,5)
    counts=np.bincount(all_epochs,minlength=67349)
    assert (counts[train]==5).all() and (counts[valid]==0).all()
    for offset in (0,999,61999):
        idx=train[offset]
        np.testing.assert_array_equal(np.where(all_epochs==idx)[0]//1000,offset//1000+62*np.arange(5))
    stats=json.loads((RESULTS/'token_lengths.json').read_text())
    assert stats['split_sha256']==file_hash(RESULTS/'split.npz')
    assert not stats['official_validation_used']


@pytest.mark.parametrize('length',[32,64,128])
def test_padding_and_local_data(length):
    train,valid=datasets(length)
    assert len(train)==62000 and len(valid)==5349
    assert train.tensors[0].shape==(62000,length)
    ids,mask,segments,labels=train.tensors
    assert ((mask==0)==(ids==0)).all()
    assert not segments.any() and labels.min()==0 and labels.max()==1


def test_locked_protocol():
    with pytest.raises(AssertionError): Config(physical_batch_size=30)
    with pytest.raises(AssertionError): Config(total_steps=250)
    with pytest.raises(AssertionError): Config(max_length=256)
    for b in (4,8,20,40,50,100,125,200,250,500,1000):
        cfg=Config(physical_batch_size=b)
        assert cfg.protocol()['privacy']['b_participation']==62 and 1000%b==0


def test_correlated_noise_retains_innovations():
    import torch
    from exp8a.noise import BandInvMFNoise
    parameter=torch.nn.Parameter(torch.zeros(5))
    d=np.array([1.,-.3,.1,-.02])
    source=BandInvMFNoise([parameter],d,2.,8,77)
    generator=torch.Generator().manual_seed(77)
    history=[]
    for step in range(8):
        z=torch.randn(5,generator=generator)*2.
        history.insert(0,z)
        expected=sum(float(a)*b for a,b in zip(d,history))
        torch.testing.assert_close(source.next()[0],expected,rtol=2e-6,atol=2e-6)
        assert source.step_count==step+1 and len(source.history)<=3


def test_pretrained_parameters_random_head_and_evaluation():
    import torch
    from torch.utils.data import TensorDataset
    from exp8a.model import make_model, seed_all, digest
    from exp8a.config import MODEL_PATH, CHECKPOINT_SHA256
    from exp8a.train import evaluate
    assert file_hash(MODEL_PATH/'pytorch_model.bin')==CHECKPOINT_SHA256
    seed_all(123)
    model=make_model()
    state=torch.load(MODEL_PATH/'pytorch_model.bin',weights_only=True)
    for name,p in model.bert.named_parameters():
        torch.testing.assert_close(p,state['bert.'+name],rtol=0,atol=0)
        assert p.requires_grad
    head=digest(model.classifier)
    seed_all(124)
    assert digest(make_model().classifier)!=head
    model.eval()
    ids=torch.tensor([[101,2023,2003,102,0,0],[101,2204,102,0,0,0],[101,2919,102,0,0,0]])
    mask=(ids!=0).long(); segments=torch.zeros_like(ids); labels=torch.tensor([0,1,0])
    ds=TensorDataset(ids,mask,segments,labels)
    result=evaluate(model,ds,torch.device('cpu'),batch_size=2)
    with torch.no_grad():
        logits=model((ids,mask,segments))
        expected_loss=torch.nn.functional.cross_entropy(logits,labels).item()
        expected_accuracy=(logits.argmax(1)==labels).float().mean().item()
    assert result['examples']==3 and result['loss']==pytest.approx(expected_loss,abs=2e-6)
    assert result['accuracy']==pytest.approx(expected_accuracy,abs=2e-6)
