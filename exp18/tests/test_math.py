import unittest,numpy as np,torch
from exp18.optimizer import adaptive,Update
from exp18.matrices import load,audit,workload
from exp18.noise import OnlineNoise
from exp18.privacy import target_mu,epsilon_from_mu
from exp18.search import coarse,trials
from exp18.model import CNN
from opacus import GradSampleModule
class MathTests(unittest.TestCase):
    def test_alpha(self):
        d=torch.tensor([.25]);v=torch.tensor([1/16])
        a,r=adaptive(torch.tensor([.5]),d,v);self.assertAlmostEqual(a,.5,places=5);self.assertGreaterEqual(float(r.square().sum()),float(v.sum()))
        self.assertEqual(adaptive(torch.tensor([1.]),d,v)[0],1)
        self.assertEqual(adaptive(torch.tensor([.125]),d,v)[0],0)
    def test_update(self):
        model=torch.nn.Linear(2,1,bias=False);o=Update(model,'stpex2',.01,.001);start=model.weight.detach().flatten().clone();q=torch.tensor([.2,.4])
        self.assertTrue(torch.equal(o.d,torch.ones(2)));a=o.step(q)
        torch.testing.assert_close(model.weight.flatten(),start-.01*q);torch.testing.assert_close(o.v,.001*q*q);self.assertEqual(a,1)
    def test_matrix_noise(self):
        for name in ('sgd','momentum','momentum-bias'):
            m=load(name);audit(m['A'],m['C'],m['D']);self.assertTrue(m['metadata']['first_order_pass'])
            p=torch.zeros(20);noise=OnlineNoise([p],m,'b2lt',1,162,123);actual=torch.stack([noise.next()[0] for _ in range(162)])
            rng=torch.Generator().manual_seed(123);z=torch.stack([torch.randn(20,generator=rng) for _ in range(162)])
            torch.testing.assert_close(actual,torch.tensor(m['D'],dtype=torch.float32)@z,atol=2e-6,rtol=2e-5)
    def test_workload(self):
        q=np.random.default_rng(1).normal(size=162);u=0;path=0;expected=[]
        for t,x in enumerate(q,1):u=.9*u+.1*x;path+=u/(1-.9**t);expected.append(path)
        np.testing.assert_allclose(workload('momentum-bias')@q,expected,atol=1e-12)
    def test_gdp_grid(self):
        self.assertAlmostEqual(epsilon_from_mu(target_mu(3,1e-5),1e-5),3)
        self.assertEqual(sum(map(len,coarse().values())),170);self.assertEqual(len(trials(coarse(),'coarse',[20261101])),340)
    def test_per_sample(self):
        torch.manual_seed(2);base=CNN();model=GradSampleModule(base,loss_reduction='sum');x=torch.randn(3,1,28,28);y=torch.tensor([1,3,5]);torch.nn.functional.cross_entropy(model(x),y,reduction='sum').backward()
        actual=torch.cat([p.grad_sample.reshape(3,-1) for p in base.parameters()],1).clone();model.remove_hooks()
        expected=[]
        for i in range(3):
            base.zero_grad();torch.nn.functional.cross_entropy(base(x[i:i+1]),y[i:i+1]).backward();expected.append(torch.cat([p.grad.flatten() for p in base.parameters()]))
        torch.testing.assert_close(actual,torch.stack(expected),atol=1e-6,rtol=1e-4)
        d=torch.rand(actual.shape[1])+.1;scaled=actual/d;clipped=scaled*(.5/scaled.norm(dim=1)).clamp_max(1)[:,None]
        self.assertTrue(bool((clipped.norm(dim=1)<=.500001).all()))
if __name__=='__main__':unittest.main()
