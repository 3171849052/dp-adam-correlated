import numpy as np
import pytest
import torch
from exp10.optim import BlockAdam,PreconditionerDiagnostics
from exp10.clipping import LogicalBatch
@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_b1_pytorch_adam_equivalence(device,dtype):
    gen=torch.Generator().manual_seed(731)
    original=[torch.randn(11,7,generator=gen,dtype=dtype),torch.randn(19,generator=gen,dtype=dtype)]
    a=[torch.nn.Parameter(v.to(device).clone()) for v in original];b=[torch.nn.Parameter(v.to(device).clone()) for v in original]
    standard=torch.optim.Adam(a,lr=.005,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
    block=BlockAdam(b,.005,1)
    for step in range(53):
        for x,y in zip(a,b):
            g=torch.randn(x.shape,generator=gen,dtype=dtype).to(device)*(1 if step%3 else .001)
            x.grad=g.clone();y.grad=g.clone()
        standard.step();block.step()
        for x,y in zip(a,b):
            torch.testing.assert_close(x,y,rtol=2e-6 if dtype==torch.float32 else 2e-12,atol=2e-7 if dtype==torch.float32 else 2e-14)
            for field in ('exp_avg','exp_avg_sq'): torch.testing.assert_close(standard.state[x][field],block.state[y][field],rtol=0,atol=0)
@pytest.mark.parametrize('b',[2,5,10,25])
def test_updated_first_v_block_boundary_and_incomplete_block(b):
    p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64));opt=BlockAdam([p],.003,b)
    m=np.zeros(2);v=np.zeros(2);theta=np.array([1.,2.]);frozen=None
    for t in range(1,2*b+2):
        g=np.array([t*.1,(-1.)**t*2]);m=.9*m+.1*g;v=.999*v+.001*g*g
        if (t-1)%b==0: frozen=1/(np.sqrt(v/(1-.999**t))+1e-8)
        theta-=.003*frozen*m/(1-.9**t)
        p.grad=torch.tensor(g);opt.step();state=opt.state[p]
        np.testing.assert_allclose(p.detach().numpy(),theta,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(state['exp_avg'].numpy(),m,rtol=1e-13)
        np.testing.assert_allclose(state['exp_avg_sq'].numpy(),v,rtol=1e-13)
        np.testing.assert_allclose(state['block_denominator'].numpy(),1/frozen,rtol=1e-13)
        assert state['block_start']==((t-1)//b)*b+1
    diagnostics=PreconditionerDiagnostics().record(opt)
    assert diagnostics['block_age']==0 and diagnostics['denominator_lag_relative']==pytest.approx(0,abs=1e-14)
def test_one_noise_innovation_per_logical_batch():
    model=torch.nn.Linear(3,2);opt=BlockAdam(model.parameters(),.002,2)
    class Noise:
        count=0
        def next(self):
            self.count+=1;return [torch.full_like(p,2.) for p in model.parameters()]
    noise=Noise();logical=LogicalBatch(model,opt,4,1000,noise)
    for step in range(5):
        for micro in range(4):
            logical.begin_microbatch()
            for p in model.parameters(): p.grad=torch.ones_like(p)
            assert logical.finish_microbatch()==(micro==3)
            assert noise.count==logical.optimizer_steps==step+int(micro==3)
        for p in model.parameters(): torch.testing.assert_close(p.grad,torch.full_like(p,.006))
