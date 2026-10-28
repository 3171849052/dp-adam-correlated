import copy
import types
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from exp17b.optimizer import directions, second_derivative, scalar_f, PrivateAdam
from exp17b.mechanism import privatize

# Execute only the historical pure optimizer file, avoiding exp17 package side effects.
old=types.ModuleType('historical_optimizer')
exec(compile(Path('exp17/optimizer.py').read_text(),'exp17/optimizer.py','exec'),old.__dict__)

@pytest.mark.parametrize('r',[0.,1.,2.8312235256706146])
def test_alpha_zero_and_state(r):
    x=torch.tensor([-.3,.1,0.],dtype=torch.float64);m=x*.2;v=x.square()+.1
    ref=directions('smooth',x,m,v,10,r,.04,0)
    for method in ['lc','gh']:
        out=directions(method,x,m,v,10,r,.04,0)
        for a,b in zip(ref,out): assert torch.equal(a,b)

@pytest.mark.parametrize('t',[1,2,10,50,250])
def test_derivative_and_historical(t):
    x=torch.tensor([-.3,.1,0.],dtype=torch.float64,requires_grad=True)
    m=torch.ones_like(x)*(.0 if t==1 else .2);v=torch.ones_like(x)*(.0 if t==1 else .1)
    r,nu2=2.8312235256706146,.04
    base,c,nm,nv=directions('lc',x,m,v,t,r,nu2,1)
    first=torch.autograd.grad(base.sum(),x,create_graph=True)[0]
    second=torch.autograd.grad(first.sum(),x)[0]
    torch.testing.assert_close(c,nu2/2*second,atol=1e-12,rtol=1e-10)
    ob,oc=old.curvature('lc',nm/(1-.9**t),nv/(1-.999**t),x,t,r,nu2)
    torch.testing.assert_close(base,ob);torch.testing.assert_close(c,oc)

def test_gh_nodes_weights_no_derivative():
    x=torch.tensor([-.5,0.,.7],dtype=torch.float64);m=torch.zeros_like(x);v=m.clone()
    with patch('exp17b.optimizer.second_derivative',side_effect=AssertionError):
        b,c,_,_=directions('gh',x,m,v,1,0.,.09,.5)
    f=lambda u:scalar_f(u,0.,1.,0.,1.,0.,.09)
    expected=(1+.5/3)*f(x)-.5/6*(f(x+(3*.09)**.5)+f(x-(3*.09)**.5))
    torch.testing.assert_close(b-c,expected)
    nodes=torch.tensor([0.,3**.5,-3**.5],dtype=torch.float64)
    weights=torch.tensor([2/3,1/6,1/6],dtype=torch.float64)
    torch.testing.assert_close(weights.sum(),torch.tensor(1.,dtype=torch.float64),atol=2e-16,rtol=0)
    torch.testing.assert_close((weights*nodes**2).sum(),torch.tensor(1.,dtype=torch.float64))
    torch.testing.assert_close((weights*nodes**4).sum(),torch.tensor(3.,dtype=torch.float64))

def test_small_noise():
    x=torch.tensor([.3,-.6],dtype=torch.float64);m=x*.4;v=x.square()+.2
    for nu2 in [1e-4,1e-6]:
        lc=directions('lc',x,m,v,10,1.,nu2,1)[1]
        gh=directions('gh',x,m,v,10,1.,nu2,1)[1]
        torch.testing.assert_close(lc,gh,atol=1e-12,rtol=nu2*10)

def test_no_extra_randomness_and_same_states():
    states=[]
    for method in ['smooth','lc','gh']:
        p=torch.nn.Parameter(torch.ones(40));opt=PrivateAdam([p],method,.01,.04,1.,.5)
        gen=torch.Generator().manual_seed(88)
        with patch('torch.randn',wraps=torch.randn) as draw:
            for _ in range(3):
                privatize([p],[torch.ones_like(p)],1.,gen)
                before=torch.get_rng_state().clone();opt.step()
                assert torch.equal(before,torch.get_rng_state())
            assert draw.call_count==3
        states.append(opt.state[p])
    for state in states[1:]:
        for key in ['m','v']: assert torch.equal(states[0][key],state[key])

def test_r0_matches_torch_adam_and_restore():
    p=torch.nn.Parameter(torch.ones(20));q=torch.nn.Parameter(p.detach().clone())
    ours=PrivateAdam([p],'smooth',.01,.04,0,0);ref=torch.optim.Adam([q],lr=.01,eps=1e-8)
    for _ in range(4):
        p.grad=torch.linspace(-1,1,20);q.grad=p.grad.clone();ours.step();ref.step()
    torch.testing.assert_close(p,q)
    r=torch.nn.Parameter(p.detach().clone());restored=PrivateAdam([r],'smooth',.01,.04,0,0)
    restored.load_state_dict(copy.deepcopy(ours.state_dict()))
    p.grad=torch.ones_like(p);r.grad=p.grad.clone();ours.step();restored.step()
    assert torch.equal(p,r)
