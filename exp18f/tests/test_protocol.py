import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from exp18f.common import (BASE,ROOT,BASELINE,METHODS,PARAMS,SEARCH_SEEDS,TURN_SEEDS,REVIEW_SEEDS,
    configs,read,save,calibration,make_config,terminal)
from exp18f.search import axis_candidates,critical,refinement,stability,baseline_metrics
from exp18f.report import stats,raw_comparison

class ProtocolTests(unittest.TestCase):
    def test_axis_and_fresh_seed_budgets(self):
        candidates=axis_candidates();jobs=configs('axis')
        self.assertEqual(len(candidates),27);self.assertEqual(len(jobs),108)
        self.assertEqual(sum(p==BASELINE for c,p in candidates),3)
        for cid,p in candidates:
            self.assertLessEqual(sum(p[k]!=BASELINE[k] for k in PARAMS),1)
            selected=[c for c in jobs if c['candidate']==cid]
            self.assertEqual({(c['method'],c['seed']) for c in selected},{(m,s) for m in METHODS for s in SEARCH_SEEDS})
            self.assertTrue(all(c['steps']==310 for c in selected))
        self.assertFalse(set(SEARCH_SEEDS)&set(TURN_SEEDS)|set(SEARCH_SEEDS)&set(REVIEW_SEEDS)|set(TURN_SEEDS)&set(REVIEW_SEEDS))
        self.assertEqual(108+3*len(TURN_SEEDS)*2+3*len(REVIEW_SEEDS)*2,156)

    def test_each_axis_clip_calibration(self):
        audit=read(BASE/'privacy_audit.json')['calibrations']
        self.assertEqual(len(audit),54)
        for cid,p in axis_candidates():
            for m in METHODS:
                r=calibration(make_config('axis',cid,m,SEARCH_SEEDS[0],p))
                self.assertEqual(r,audit[f'{cid}/{m}'])
                self.assertAlmostEqual(r['epsilon'],8,places=9)
                self.assertFalse(r['sampling_amplification'])
        for m in METHODS:
            a=calibration(make_config('axis','a',m,1,dict(BASELINE,clip=50)))
            b=calibration(make_config('axis','b',m,1,dict(BASELINE,clip=200)))
            self.assertAlmostEqual(b['sigma_sum']/a['sigma_sum'],4)
            self.assertEqual(a['sensitivity'],b['sensitivity'])

    def points(self,values,flags):
        return [dict(lr=v,stable=f,mean_accuracy=.8 if f else .5,
            methods={m:dict(label='stable' if f else 'numerical_unstable') for m in METHODS}) for v,f in zip(values,flags)]

    def test_monotonic_geometric_search(self):
        points=self.points([.02,.03,.04,.08],[True,True,False,False])
        info=critical(points);self.assertFalse(info['nonmonotonic'])
        self.assertEqual(info['high_side_interval'],dict(low=.03,high=.04))
        lr,mode=refinement(points,{p['lr'] for p in points})
        self.assertAlmostEqual(lr,math.sqrt(.03*.04));self.assertEqual(mode,'geometric_bisection')

    def test_nonmonotonic_local_search_and_censoring(self):
        points=self.points([.01,.02,.03,.04,.08],[True,False,True,True,False])
        info=critical(points);self.assertTrue(info['nonmonotonic']);self.assertEqual(len(info['intervals']),2)
        lr,mode=refinement(points,{p['lr'] for p in points})
        self.assertEqual(mode,'local_densification_nonmonotonic')
        self.assertTrue(any(a['lr']<lr<b['lr'] for a,b in zip(points,points[1:])))
        stable=self.points([.02,.03,.04],[True,True,True]);self.assertEqual(critical(stable)['coverage'],'right_censored')
        self.assertIsNone(critical(stable)['high_side_interval'])
        self.assertEqual(refinement(stable,{p['lr'] for p in stable})[1],'local_densification_unbracketed')

    def test_numeric_vs_predictive_and_alpha(self):
        def row(cid,status,accuracy=.8,loss=.4):
            return dict(candidate=cid,parameters=BASELINE,mean_accuracy=accuracy,
                trials=[dict(method=m,seed=1,status=status,metrics=dict(accuracy=accuracy,loss=loss),alpha=0) for m in METHODS])
        bases={m:dict(accuracy=.8,loss=.4) for m in METHODS}
        labels=stability([row('good','complete'),row('bad','complete',.5,1.2),row('nan','numerical_failure')],bases)
        self.assertTrue(labels[0]['stable'])
        self.assertEqual(labels[1]['methods'][METHODS[0]]['label'],'predictive_unstable')
        self.assertEqual(labels[2]['methods'][METHODS[0]]['label'],'numerical_unstable')
        failed=row('lr_base','numerical_failure');baseline=baseline_metrics([failed])
        self.assertEqual(baseline[METHODS[0]]['reference'],'predeclared absolute thresholds')

    def test_numeric_terminal_and_integrity(self):
        from exp18f.train import finish
        with tempfile.TemporaryDirectory(dir=BASE/'tests') as folder:
            out=Path(folder);cfg=make_config('axis','test',METHODS[0],1,BASELINE)
            save(out/'config.json',cfg);save(out/'summary.json',dict(status='numerical_failure',steps=0))
            (out/'steps.jsonl').write_text('');finish(out,'numerical_failure')
            with patch('exp18f.common.outdir',return_value=out):
                self.assertEqual(terminal(cfg,verify=True)['status'],'numerical_failure')
                (out/'steps.jsonl').write_text('corruption')
                with self.assertRaises(AssertionError):terminal(cfg,verify=True)

    def test_statistics_and_descriptive_comparison(self):
        self.assertAlmostEqual(stats([1,2,3,4,5])['sd'],np.sqrt(2.5))
        c=raw_comparison([1,2,3,4,5],[1,2,3])
        self.assertEqual(c['difference'],1)
        self.assertLess(c['ci95_low'],1);self.assertGreater(c['ci95_high'],1)
        self.assertIsNone(stats([])['mean'])

    def test_full_three_stage_orchestration_and_review_only_freeze(self):
        from exp18f import search,common
        import shutil
        original_plan=configs('axis');spec=read(BASE/'search_space.json')
        with tempfile.TemporaryDirectory(dir=BASE/'tests') as folder:
            root=Path(folder);save(root/'search_space.json',spec);save(root/'plans/axis.json',original_plan)
            def fake_calibration(cfg):return dict(clip=cfg['clip'])
            def fake_run(stage):
                for cfg in common.configs(stage):
                    out=common.outdir(cfg);out.mkdir(parents=True)
                    # Search winner is .056; review winner is .035 on new seeds.
                    acc=(.90-abs(cfg['lr']-.035)) if stage=='rounded' else (.91-abs(cfg['lr']-.056))
                    save(out/'config.json',cfg)
                    save(out/'summary.json',dict(status='complete',steps=310,pairing=dict(seed=cfg['seed']),
                        metrics=dict(accuracy=acc,loss=.4,examples=5349)))
                    save(out/'terminal.json',dict(files={}))
            with patch.object(search,'BASE',root),patch.object(common,'BASE',root),\
                patch.object(search,'calibration',fake_calibration),patch('exp18f.scheduler.run',fake_run):
                frozen=search.run_search()
                self.assertTrue(frozen['frozen']);self.assertEqual(frozen['parameters']['lr'],.035)
                self.assertEqual(frozen['seeds'],REVIEW_SEEDS)
                plans=list((root/'plans').glob('*.json'))
                self.assertEqual(sum(len(read(p)) for p in plans),156)
                for cfg in common.configs('rounded'):self.assertIn(cfg['seed'],REVIEW_SEEDS)
                self.assertEqual(len(read(root/'turning_point.json')['history']),3)
                from exp18f import report as reporting
                import json
                (root/'plots').mkdir()
                with patch.object(reporting,'BASE',root):
                    counts=reporting.report(plots=False)
                    self.assertEqual(counts['complete'],156)
                    rounded=[r for r in read(root/'search/statistics.json') if r['stage']=='rounded']
                    self.assertEqual(len(rounded),6)
                    self.assertTrue(all(r['accuracy']['n']==5 for r in rounded))
                    rows,_=reporting.planned_records()
                    plotted=[r for r in rows if r['stage']=='rounded' and r['candidate']=='rounded_1'][:2]
                    for r in plotted:
                        (root/r['directory']/'steps.jsonl').write_text(json.dumps(dict(step=1,train_loss=.5,
                            train_loss_smooth10=.5,alpha=0.,clipped_fraction=.2))+'\n')
                    reporting.make_plots(plotted)
                    self.assertEqual(len(list((root/'plots').glob('*.png'))),6)

    def scheduler_fixture(self,codes):
        from exp18f import scheduler
        settings=read(BASE/'scheduling.json');jobs=[make_config('smoke',f'test{i}',METHODS[0],i,BASELINE) for i in range(len(codes))]
        launched={};self.terminated=[]
        with tempfile.TemporaryDirectory(dir=BASE/'tests') as folder:
            root=Path(folder);(root/'logs').mkdir()
            def result(c,verify=False):
                index=jobs.index(c)
                if index not in launched or codes[index]!=0:return None
                return dict(status='numerical_failure' if index==0 else 'complete')
            class FakeProcess:
                def __init__(inner,command,**kwargs):
                    index=len(launched);inner.index=index;inner.pid=100+index;launched[index]=inner
                def poll(inner):return codes[inner.index]
                def terminate(inner):self.terminated.append(inner.pid)
                def wait(inner):return codes[inner.index]
            inventory={g:dict(uuid=settings['gpu_uuids'][str(g)],free_mib=12000) for g in settings['physical_gpus']}
            with patch.multiple(scheduler,BASE=root,configs=lambda stage:jobs,terminal=result,
                outdir=lambda c:root/c['candidate'],gpu_snapshot=lambda:inventory,process_memory=lambda:{}),\
                patch.object(scheduler,'read',return_value=settings),patch.object(scheduler.subprocess,'Popen',FakeProcess),\
                patch.object(scheduler.time,'sleep'):
                if any(codes):
                    with self.assertRaisesRegex(RuntimeError,'Infrastructure failure'):scheduler.run('smoke')
                else:self.assertEqual(len(scheduler.run('smoke')),len(jobs))
            return read(root/'logs/smoke_status.json')

    def test_scheduler_continues_after_numerical_failure(self):
        state=self.scheduler_fixture([0,0]);self.assertEqual(state['status'],'complete')
        self.assertEqual(state['numerical_failures'],1);self.assertEqual(state['completed'],1)

    def test_scheduler_stops_on_infrastructure_failure(self):
        state=self.scheduler_fixture([1,0]);self.assertEqual(state['status'],'infrastructure_failed')

if __name__=='__main__':unittest.main()
