import copy
import unittest
import torch
from exp17.mechanism import poisson_indices, privatize
from exp17.clipping import StandardGhostModule, clipped_microbatch


class MechanismTests(unittest.TestCase):
    def test_independent_membership(self):
        g=torch.Generator().manual_seed(55)
        draws=torch.zeros(10000,40)
        for i in range(len(draws)): draws[i,poisson_indices(40,.2,g)]=1
        self.assertLess(abs(draws.mean().item()-.2),.003)
        self.assertLess(abs(draws.sum(1).var().item()-6.4),.25)
        self.assertLess(abs((draws[:,0]*draws[:,1]).mean().item()-.04),.006)
        self.assertLess(abs((draws[:-1,0]*draws[1:,0]).mean().item()-.04),.006)
        self.assertEqual(len(poisson_indices(40,0,g)),0)

    def test_noise_empty_and_fixed_denominator(self):
        p=torch.nn.Parameter(torch.zeros(200000));g=torch.Generator().manual_seed(9)
        privatize([p],[torch.zeros_like(p)],2.,g)
        self.assertLess(abs(p.grad.mean().item()),.0005)
        self.assertLess(abs(p.grad.var().item()-.06**2),.00005)
        rng=g.get_state();privatize([p],[torch.full_like(p,123.)],2.,g);x=p.grad.clone()
        g.set_state(rng);privatize([p],[torch.zeros_like(p)],2.,g)
        torch.testing.assert_close(x-p.grad,torch.full_like(p,.123),atol=4e-8,rtol=1e-6)

    def test_clipping_variable_microbatches(self):
        torch.manual_seed(12)
        base=torch.nn.Sequential(torch.nn.Linear(4,7),torch.nn.Tanh(),torch.nn.Linear(7,3))
        model=StandardGhostModule(copy.deepcopy(base),.3)
        x=torch.randn(7,4);y=torch.randint(0,3,(7,));expected=[torch.zeros_like(p) for p in base.parameters()]
        for xx,yy in zip(x,y):
            base.zero_grad();torch.nn.functional.cross_entropy(base(xx[None]),yy[None]).backward()
            norm=sum(p.grad.square().sum() for p in base.parameters()).sqrt()
            for s,p in zip(expected,base.parameters()): s.add_(p.grad*min(1.,.3/norm.item()))
        actual=[torch.zeros_like(p) for p in model.parameters()]
        for lo,hi in ((0,3),(3,6),(6,7)):
            model.zero_grad();clipped_microbatch(model,x[lo:hi],y[lo:hi],.3)
            for s,p in zip(actual,model.parameters()): s.add_(p.grad)
        for a,b in zip(actual,expected): torch.testing.assert_close(a,b,atol=2e-7,rtol=1e-5)
        model.remove_hooks()
