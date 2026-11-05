import copy
import unittest
import numpy as np
import torch
from transformers import BertConfig, BertModel
from exp18d.common import (BASE, GROUPS, SEEDS, configs, load_matrix, read, BATCH)
from exp18d.model import Classifier
from exp18d.clipping import StandardGhostModule, ScaledGhostModule
from exp18d.train import clipped_sum
from exp18.optimizer import Update, adaptive
from exp18.noise import OnlineNoise
from exp18.privacy import epsilon_from_mu


class ExperimentTests(unittest.TestCase):
    def test_plan_privacy_and_frozen_sources(self):
        plan = configs()
        self.assertEqual(len(plan),60)
        self.assertEqual(len({(c['method'],c['seed']) for c in plan}),60)
        self.assertEqual(SEEDS,list(range(20261111,20261121)))
        for p in read(BASE / 'privacy_audit.json')['methods'].values():
            self.assertAlmostEqual(epsilon_from_mu(p['clip']*p['sensitivity']/p['sigma_sum'],1e-5),8)
            self.assertFalse(p['sampling_amplification'])
        self.assertEqual(read(BASE/'hyperparameters.json')['methods']['stpex2-iid'],dict(lr=.08,clip=100,eps=.005))

    def test_online_noise_matches_dense_full_horizon(self):
        for name in ('sgd','momentum','momentum-bias'):
            m = load_matrix(name)
            noise = OnlineNoise([torch.zeros(7)],m,'b2lt',1,310,123)
            actual = torch.stack([noise.next()[0] for _ in range(310)])
            rng = torch.Generator().manual_seed(123)
            z = torch.stack([torch.randn(7,generator=rng) for _ in range(310)])
            torch.testing.assert_close(actual,torch.tensor(m['D'],dtype=torch.float32)@z,atol=3e-6,rtol=3e-5)
            np.testing.assert_allclose(m['C']@m['D'],np.eye(310),atol=1e-8)
            th=np.array(m['metadata']['theta']); om=np.array(m['metadata']['omega'])
            self.assertTrue(np.all(np.diff(th)>0) and np.all(om<0))
            self.assertGreater(1+(om/(1-th)).sum(),0)
            self.assertEqual(np.count_nonzero(m['d'][62:]),0)

    def test_ghost_clipped_sum_against_independent_bert_autograd(self):
        torch.set_num_threads(2); torch.manual_seed(42)
        cfg = BertConfig(vocab_size=19,hidden_size=12,num_hidden_layers=1,num_attention_heads=3,
            intermediate_size=24,max_position_embeddings=16,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
        cfg._attn_implementation='eager'
        base = Classifier(BertModel(cfg))
        ids=torch.tensor([[1,2,2,0,0],[3,3,4,3,0],[1,1,1,1,0]])
        inputs=(ids,(ids!=0).long(),torch.zeros_like(ids)); y=torch.tensor([0,1,1])
        for group in ('adam','stpex2'):
            model=(ScaledGhostModule if group=='stpex2' else StandardGhostModule)(copy.deepcopy(base),.7)
            opt=Update(model,group,.01,.005)
            if group=='stpex2': opt.d=torch.rand_like(opt.d)*2+.1
            expected=torch.zeros_like(opt.d)
            for i in range(3):
                base.zero_grad()
                torch.nn.functional.cross_entropy(base(tuple(x[i:i+1] for x in inputs)),y[i:i+1]).backward()
                g=torch.cat([p.grad.flatten() for p in base.parameters()])
                if group=='stpex2': g=g/opt.d
                expected.add_(g*(.7/g.norm()).clamp_max(1))
            actual,_,_=clipped_sum(model,inputs,y,opt,.7)
            torch.testing.assert_close(actual,expected,atol=3e-6,rtol=6e-5)
            model.remove_hooks()

    def test_updates_and_adaptive_constraint(self):
        d=torch.tensor([.25]); v=torch.tensor([1/16])
        a,r=adaptive(torch.tensor([.5]),d,v)
        self.assertAlmostEqual(a,.5,places=5)
        self.assertGreaterEqual(float(r.square().sum()),float(v.sum()))
        for group in ('adam','momentum','stpex2'):
            model=torch.nn.Linear(2,1,bias=False); opt=Update(model,group,.01,.005)
            w=model.weight.detach().flatten().clone(); u=torch.zeros(2); v=torch.zeros(2); d=torch.ones(2)
            for t in range(1,5):
                q=torch.tensor([.2/t,-.4*t]); u=.9*u+.1*q
                if group=='stpex2':
                    old=v/(1-.999**(t-1)) if t>1 else v
                    a,r=adaptive(q,d,old); v=.999*v+.001*r*r
                    d=(v/(1-.999**t)).sqrt()+.005; update=u/(1-.9**t)
                elif group=='adam':
                    v=.999*v+.001*q*q; update=(u/(1-.9**t))/((v/(1-.999**t)).sqrt()+1e-8)
                else: update=u
                w=w-.01*update; opt.step(q)
                torch.testing.assert_close(model.weight.flatten(),w)
                torch.testing.assert_close(opt.d,d)

if __name__=='__main__': unittest.main()
