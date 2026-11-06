import copy
import unittest
import numpy as np
import torch
from transformers import BertConfig, BertModel
from exp18e.common import (BASE, GROUPS, SEEDS, configs, load_matrix, read, BATCH)
from exp18e.model import Classifier
from exp18e.clipping import StandardGhostModule, ScaledGhostModule
from exp18e.train import clipped_sum
from exp18e.optimizer import Update, adaptive
from exp18e.noise import OnlineNoise
from exp18e.privacy import epsilon_from_mu


class ExperimentTests(unittest.TestCase):
    def test_plan_privacy_and_frozen_sources(self):
        from unittest.mock import patch
        from exp18e.common import FIXED,HISTORICAL,calibration,make_config
        with patch('exp18e.common.read',return_value={'parameters':HISTORICAL}):
            plan=configs()
        self.assertEqual(len(plan),30)
        self.assertEqual(len({(c['method'],c['seed']) for c in plan}),30)
        self.assertEqual(SEEDS,[20261111,20261112,20261113,20261114,20261115])
        for seed in SEEDS:
            self.assertEqual(sum(c['seed']==seed for c in plan),6)
        self.assertEqual(FIXED['adam-b2lt-sgd']['lr'],.01)
        self.assertEqual(FIXED['adam-iid']['clip'],1)
        self.assertEqual(FIXED['momentum-iid']['clip'],1)
        for c in plan[:6]:
            p=calibration(c)
            self.assertAlmostEqual(epsilon_from_mu(p['clip']*p['sensitivity']/p['sigma_sum'],1e-5),8)
            self.assertFalse(p['sampling_amplification'])
        for m in GROUPS['stpex2']:
            a=calibration(make_config('coarse',m,20261102,dict(lr=.001,clip=3,eps=.001)))
            b=calibration(make_config('coarse',m,20261102,dict(lr=.001,clip=100,eps=.001)))
            self.assertAlmostEqual(b['sigma_sum']/a['sigma_sum'],100/3)

    def test_search_coordinates_boundaries_and_ranking(self):
        from exp18e.search import coarse_candidates,fine_candidates,boundary_candidates,scores
        from exp18e.common import HISTORICAL,candidate_id,make_config
        from unittest.mock import patch
        coarse=coarse_candidates()
        self.assertEqual(len(coarse),121)
        self.assertIn(HISTORICAL,coarse)
        winners=[dict(parameters=coarse[0]),dict(parameters=coarse[-1])]
        fine=fine_candidates(winners,coarse)
        self.assertLessEqual(len(fine),14)
        self.assertGreater(len(fine),0)
        for p in fine:
            self.assertTrue(any(sum(p[k]!=w['parameters'][k] for k in p)==1 for w in winners))
            self.assertNotIn(candidate_id(p),{candidate_id(c) for c in coarse})
        boundary=boundary_candidates(winners,coarse+fine)
        self.assertLessEqual(len(boundary),2)
        self.assertEqual(boundary[0]['lr'],coarse[0]['lr']*.5)
        # Equal paired accuracy: lower validation loss wins, independent of method order.
        a,b=coarse[:2]
        jobs=[make_config('coarse',m,20261102,p) for p in (a,b) for m in GROUPS['stpex2']]
        def result(cfg):
            return dict(pairing={'seed':cfg['seed']},evaluation='internal_validation',metrics=dict(examples=5349,accuracy=.7,
                loss=.8 if cfg['eps']==a['eps'] else .6))
        with patch('exp18e.search.configs',return_value=jobs),patch('exp18e.search.completed',side_effect=result):
            ranked=scores(['coarse'])
        self.assertEqual(ranked[0]['parameters'],b)
        self.assertEqual(len(ranked[0]['trials']),2)

    def test_all_candidate_calibrations(self):
        from exp18e.search import coarse_candidates
        from exp18e.common import candidate_id
        audit=read(BASE/'search/coarse_calibration.json')
        self.assertEqual(len(audit),121)
        for params in coarse_candidates():
            for m in GROUPS['stpex2']:
                p=audit[candidate_id(params)][m]
                self.assertEqual(p['clip'],params['clip'])
                self.assertAlmostEqual(epsilon_from_mu(p['clip']*p['sensitivity']/p['sigma_sum'],1e-5),8)

    def test_statistics_use_sample_sd_and_student_t(self):
        from exp18e.report import stats
        from scipy.stats import t
        s=stats([1,2,3,4,5])
        self.assertAlmostEqual(s['mean'],3)
        self.assertAlmostEqual(s['sd'],np.sqrt(2.5))
        self.assertAlmostEqual(s['ci95_high']-3,t.ppf(.975,4)*np.sqrt(.5))
        self.assertIsNone(stats([])['mean'])

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
