import copy
import unittest
import torch
from exp17.optimizer import PrivateAdam, curvature, smooth


class OptimizerTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(71)

    def test_adam(self):
        a=torch.nn.Parameter(torch.randn(9,dtype=torch.float64));b=torch.nn.Parameter(a.detach().clone())
        ours=PrivateAdam([a],'adam',.003,.02);ref=torch.optim.Adam([b],lr=.003,eps=1e-8,weight_decay=0)
        for _ in range(20):
            a.grad=torch.randn_like(a);b.grad=a.grad.clone();ours.step();ref.step()
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_bc_negative(self):
        p=torch.nn.Parameter(torch.zeros(3,dtype=torch.float64));o=PrivateAdam([p],'bc',.01,.5,.01)
        x=torch.tensor([0.,.1,2.],dtype=torch.float64);p.grad=x;o.step()
        torch.testing.assert_close(p,-.01*x/(x.square()-.5).clamp_min(.005).sqrt(),rtol=1e-12,atol=1e-12)

    def test_lc_autodiff(self):
        for t in (1,2,13):
            x=torch.randn(5,dtype=torch.float64,requires_grad=True)
            mp=torch.randn_like(x) if t>1 else torch.zeros_like(x)
            vp=torch.rand_like(x) if t>1 else torch.zeros_like(x)
            M=(.9*mp+.1*x)/(1-.9**t);V=(.999*vp+.001*x*x)/(1-.999**t)
            f=M/((V+.03).sqrt()+1e-8)
            first=torch.autograd.grad(f.sum(),x,create_graph=True)[0]
            second=torch.autograd.grad(first.sum(),x)[0]
            _,corr=curvature('lc',M,V,x,t,.3,.1)
            torch.testing.assert_close(corr/.05,second,rtol=1e-10,atol=1e-10)

    def test_hc_autodiff_and_recurrence(self):
        for t in (1,2,7):
            xs=torch.randn(t,4,dtype=torch.float64,requires_grad=True)
            powers=torch.arange(t-1,-1,-1,dtype=torch.float64)[:,None]
            a=.1*.9**powers/(1-.9**t);b=.001*.999**powers/(1-.999**t)
            M=(a*xs).sum(0);V=(b*xs.square()).sum(0)
            R=torch.zeros(4,dtype=torch.float64);H=R.clone()
            for x in xs:
                R=.9*.999*R+x;H=.999**2*H+x*x
            torch.testing.assert_close(.1*.001/((1-.9**t)*(1-.999**t))*R,(a*b*xs).sum(0))
            torch.testing.assert_close(.001**2/(1-.999**t)**2*H,(b*b*xs*xs).sum(0))
            f=M/((V+.03).sqrt()+1e-8)
            grad=torch.autograd.grad(f.sum(),xs,create_graph=True)[0]
            lap=sum(torch.autograd.grad(grad[s].sum(),xs,retain_graph=True)[0][s] for s in range(t))
            _,corr=curvature('hc',M,V,xs[-1],t,.3,.1,R,H)
            torch.testing.assert_close(corr/.05,lap,rtol=1e-10,atol=1e-10)

    def test_stability_and_checkpoint(self):
        for method in ('adam','bc','lc','hc'):
            p=torch.nn.Parameter(torch.zeros(4));o=PrivateAdam([p],method,.001,1e-6,.001)
            for val in (0.,1e-30,1e-20,.01): p.grad=torch.full_like(p,val);o.step()
            q=torch.nn.Parameter(p.detach().clone());other=PrivateAdam([q],method,.001,1e-6,.001)
            other.load_state_dict(copy.deepcopy(o.state_dict()))
            for _ in range(3):
                p.grad=torch.randn_like(p);q.grad=p.grad.clone();o.step();other.step()
                torch.testing.assert_close(p,q,rtol=0,atol=0)
            self.assertTrue(torch.isfinite(p).all())
            if method=='hc':
                for k in ('R','H'): torch.testing.assert_close(o.state[p][k],other.state[q][k])

if __name__=='__main__': unittest.main()
