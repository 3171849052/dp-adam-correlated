"""Independent mathematical references for the original Exp16 STP pipeline."""
import copy
import io
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from transformers import BertConfig,BertModel
from exp18g.common import *
from exp18g.model import Classifier
from exp18g.clipping import ScaledGhostModule,freeze_adam_scale,clipped_microbatch
from exp18g.optimizer import make_optimizer,LogicalBatch
from exp18g.noise import OnlineNoise

class MathTests(unittest.TestCase):
    def test_scaled_bert_norms_and_clipped_sum(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        cfg=BertConfig(vocab_size=19,hidden_size=12,num_hidden_layers=1,num_attention_heads=3,
            intermediate_size=24,max_position_embeddings=16,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
        cfg._attn_implementation='eager';base=Classifier(BertModel(cfg))
        ids=torch.tensor([[1,2,2,0,0],[3,3,4,3,0],[1,1,1,1,0]])
        inputs=(ids,(ids!=0).long(),torch.zeros_like(ids));y=torch.tensor([0,1,1])
        model=ScaledGhostModule(copy.deepcopy(base),.7)
        opt=make_optimizer(model,SimpleNamespace(lr=.01))
        # Different coordinates have different vhat; no separable layer approximation.
        for p in model.parameters():
            opt.state[p]=dict(step=torch.tensor(4.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.rand_like(p)*.01)
        freeze_adam_scale(opt,.001)
        expected=[torch.zeros_like(p) for p in base.parameters()];norms=[]
        for i in range(3):
            base.zero_grad()
            torch.nn.functional.cross_entropy(base(tuple(x[i:i+1] for x in inputs)),y[i:i+1]).backward()
            gs=[p.grad*s._logical_scale for p,s in zip(base.parameters(),model.parameters())]
            norm=torch.cat([g.flatten() for g in gs]).norm();norms.append(norm)
            coef=(.7/norm).clamp_max(1)
            for e,g in zip(expected,gs):e.add_(g*coef)
        clipped_microbatch(model,inputs,y,.7)
        torch.testing.assert_close(model.last_norms,torch.stack(norms),atol=3e-6,rtol=6e-5)
        for p,e in zip(model.parameters(),expected):
            torch.testing.assert_close(p.grad*p._logical_scale,e,atol=3e-6,rtol=6e-5)
        model.remove_hooks()

    def test_restore_and_adam_against_manual_multistep_reference(self):
        torch.manual_seed(12);model=torch.nn.Linear(3,1,bias=False)
        opt=make_optimizer(model,SimpleNamespace(lr=.01))
        zs=[torch.tensor([[.2,-.7,.4]])*(t+1) for t in range(4)]
        class FixedNoise:
            def __init__(self):self.t=0
            def next(self):
                z=zs[self.t];self.t+=1;return [z.clone()]
        logical=LogicalBatch(model,opt,1,logical_size=3,noise=FixedNoise(),scaled=True,eps_scale=.001)
        w=model.weight.detach().clone();u=torch.zeros_like(w);v=torch.zeros_like(w)
        samples=torch.tensor([[1.,2.,-3.],[-.2,.4,.8],[.5,.6,.7]])
        for t in range(1,5):
            logical.begin_microbatch()
            vhat=v/(1-.999**(t-1)) if t>1 else v
            scale=1/(vhat.sqrt()+.001)
            torch.testing.assert_close(model.weight._logical_scale,scale)
            scaled=samples*scale
            coefs=(500/scaled.norm(dim=1)).clamp_max(1)
            clipped=(samples*coefs[:,None]).sum(0,keepdim=True)
            restored=(clipped*scale+zs[t-1])/scale/3
            model.weight.grad=clipped.clone()
            self.assertTrue(logical.finish_microbatch())
            torch.testing.assert_close(model.weight.grad,restored)
            u=.9*u+.1*restored;v=.999*v+.001*restored.square()
            w=w-.01*(u/(1-.9**t))/((v/(1-.999**t)).sqrt()+1e-8)
            torch.testing.assert_close(model.weight,w,atol=1e-7,rtol=1e-6)
            torch.testing.assert_close(opt.state[model.weight]['exp_avg_sq'],v)
        self.assertEqual(logical.optimizer_steps,4)
        self.assertEqual(opt.param_groups[0]['eps'],1e-8)

    def test_noise_full_horizon_and_covariance(self):
        m=load_matrix();sigma=1.7
        noise=OnlineNoise([torch.zeros(13)],m,'b2lt',sigma,T,123)
        actual=torch.stack([noise.next()[0] for _ in range(T)])
        rng=torch.Generator().manual_seed(123)
        z=torch.stack([torch.randn(13,generator=rng)*sigma for _ in range(T)])
        torch.testing.assert_close(actual,torch.tensor(m['D'],dtype=torch.float32)@z,atol=3e-6,rtol=3e-5)
        # Columns are independent innovations: the exact linear map implies covariance sigma² DDᵀ.
        cov=sigma**2*m['D']@m['D'].T
        np.testing.assert_allclose(m['C']@cov@m['C'].T,sigma**2*np.eye(T),atol=1e-10)
        # Empirical temporal covariance at representative lags, including expiry P=62.
        rng=np.random.default_rng(123)
        idx=np.array([0,1,30,61,62,63,100,309]);N=100000
        draws=m['D'][idx]@rng.standard_normal((T,N))
        empirical=draws@draws.T/N
        np.testing.assert_allclose(empirical,(m['D']@m['D'].T)[np.ix_(idx,idx)],atol=.014,rtol=.025)

    def test_privacy_and_five_frozen_runs(self):
        self.assertEqual(len(configs()),5)
        self.assertEqual([c['seed'] for c in configs()],SEEDS)
        for c in configs():
            self.assertEqual({k:c[k] for k in PARAMS},PARAMS)
            a=calibration(c)
            self.assertAlmostEqual(a['epsilon'],8,places=9)
            self.assertEqual(a['clip'],500);self.assertFalse(a['sampling_amplification'])
        self.assertEqual((BASE/'clipping.py').read_bytes(),(ROOT/'exp16a/clipping.py').read_bytes())
        verify_seal()

def main():
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(MathTests)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    output=stream.getvalue();print(output,flush=True)
    save(BASE/'acceptance/unit_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),output=output,time=now()))
    assert result.wasSuccessful()
if __name__=='__main__':main()
