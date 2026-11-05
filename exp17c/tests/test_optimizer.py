import copy
import pytest
import torch
from exp17c.optimizer import second_derivative,PrivateAdam,directions

@pytest.mark.parametrize('step',[1,2,10,50,250])
def test_analytic_autograd(step):
    u=torch.tensor([-0.3,-1e-8,1e-9,.2],dtype=torch.float64,requires_grad=True)
    m=torch.full_like(u,.12 if step>1 else 0)
    v=torch.full_like(u,.02 if step>1 else 0)
    A,B=.9*m/(1-.9**step),.1/(1-.9**step)
    K,D=.999*v/(1-.999**step),.001/(1-.999**step)
    f=(A+B*u)/((K+D*u*u).sqrt()+1e-8)
    first=torch.autograd.grad(f.sum(),u,create_graph=True)[0]
    second=torch.autograd.grad(first.sum(),u)[0]
    torch.testing.assert_close(second_derivative(u,m,v,step),second,rtol=1e-7,atol=2e-12)

def test_zero_and_tiny():
    u=torch.tensor([0.,1e-30,-1e-30],dtype=torch.float64)
    d=second_derivative(u,u*0,u*0,1)
    assert d[0]==0 and torch.isfinite(d).all()
    assert d[1]<0 and d[2]>0

def test_adam_and_resume():
    p=torch.nn.Parameter(torch.ones(20));q=torch.nn.Parameter(p.detach().clone())
    opt=PrivateAdam([p],'lc',.001,.01,0.,0.)
    ref=torch.optim.Adam([q],lr=.001,eps=1e-8)
    for i in range(4):
        p.grad=torch.linspace(-1,1,20)+i*.01;q.grad=p.grad.clone()
        opt.step();ref.step()
    torch.testing.assert_close(p,q)
    r=torch.nn.Parameter(p.detach().clone());other=PrivateAdam([r],'lc',.001,.01,0.,2.)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    p.grad=torch.ones_like(p);r.grad=p.grad.clone();opt.step();other.step()
    assert torch.equal(p,r)

def test_alpha_and_frozen_states():
    x=torch.tensor([-.2,1e-8,.3]);m=x*.1;v=x*x
    outputs=[directions('lc',x,m,v,2,0,.01,a) for a in [0,1e-14,2.]]
    for out in outputs[1:]:
        assert torch.equal(out[2],outputs[0][2]) and torch.equal(out[3],outputs[0][3])
    torch.testing.assert_close(outputs[2][1],outputs[1][1]*(2/1e-14))
