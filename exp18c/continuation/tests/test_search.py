import unittest
from copy import deepcopy
from unittest.mock import patch

from exp18c.continuation import BASE, ROOT
from exp18c.continuation.common import GRID, GROUPS, SEEDS, next_rates, read, sha, plan
from exp18c.continuation.search import aggregate, assess, best_point


class SearchTests(unittest.TestCase):
    def test_round_numbers_and_fixed_first_round(self):
        self.assertEqual(next_rates(.06),[.08,.1,.15])
        self.assertEqual(next_rates(.08),[.1,.15,.2])
        self.assertEqual(next_rates(.8),[1.,1.5,2.])
        self.assertEqual(next_rates(2),[3.,4.,5.])
        self.assertEqual(next_rates(.2),[.3,.4,.5])
        self.assertEqual(next_rates(8),[10.,15.,20.])
        self.assertEqual(GRID,read(BASE/'rounds/001/plan.json')['grid'])

    def test_only_lr_changes_and_same_gdp(self):
        old=read(ROOT/'exp18c/protocol.json')
        new=read(BASE/'protocol.json')
        for key in ('split','split_seed','seeds','epochs','steps','physical_batch','logical_batch','precision',
                    'epsilon','delta','privacy','algorithm_version','pairing','B2LT'):
            self.assertEqual(old[key],new[key])
        self.assertEqual(read(BASE/'privacy_audit.json')['methods'],read(ROOT/'exp18c/privacy_audit.json')['methods'])
        for cfg in plan():
            self.assertEqual(cfg['clip'],old['grid'][cfg['group']]['clip'])
            self.assertEqual(cfg['eps'],.005 if cfg['group']=='stpex2' else 1e-8)

    def test_one_noisy_dip_does_not_stop_search(self):
        points=[]
        for group in GROUPS:
            for view in ('iid','b2lt','average'):
                for lr,acc in ((1.,.9),(2.,.8999)):
                    points.append(dict(group=group,view=view,lr=lr,valid=True,
                        validation_accuracy=acc,validation_loss=.2,diverged_trials=0))
        result=assess(points)
        self.assertFalse(any(r['passed'] for r in result.values()))
        for group in GROUPS:
            for view in ('iid','b2lt','average'):
                points.append(dict(group=group,view=view,lr=3.,valid=True,
                    validation_accuracy=.88,validation_loss=.3,diverged_trials=0))
        self.assertTrue(all(r['passed'] for r in assess(points).values()))
        # A new best at the high boundary revokes the stopping condition.
        points[-1]['validation_accuracy']=.95
        self.assertFalse(assess(points)['stpex2']['passed'])

    def test_equal_weighting_and_selection_tie_loss(self):
        rows=[]
        for cfg in plan():
            cfg=deepcopy(cfg)
            lr_index=GRID[cfg['group']].index(cfg['lr'])
            # Every LR ties on equal-noise accuracy; IID alone prefers high LR.
            acc=[.3,.5,.7][lr_index] if cfg['method'].endswith('-iid') else [.7,.5,.3][lr_index]
            rows.append(dict(config=cfg,status='complete',metrics=dict(accuracy=acc,loss=[.4,.1,.3][lr_index]),source_experiment='synthetic'))
        points=aggregate(rows)
        for group in GROUPS:
            best=best_point([p for p in points if p['group']==group and p['view']=='average'])
            self.assertEqual(best['lr'],GRID[group][1])
        # Divergence excludes a candidate without inventing a numeric score.
        rows[0]['status']='diverged'
        rows[0]['metrics']=None
        points=aggregate(rows)
        point=next(p for p in points if p['group']=='adam' and p['view']=='average' and p['lr']==.08)
        self.assertFalse(point['valid'])
        self.assertIsNone(point['validation_accuracy'])

    def test_new_assets_match_frozen_sources(self):
        originals=read(ROOT/'exp18b/initializations.json')
        copied=read(BASE/'initializations.json')
        for seed in SEEDS:
            self.assertEqual(sha(ROOT/originals[str(seed)]['path']),sha(ROOT/copied[str(seed)]['path']))
        for p in (BASE/'matrices').iterdir():
            self.assertEqual(sha(p),sha(ROOT/'exp18c/matrices'/p.name))
