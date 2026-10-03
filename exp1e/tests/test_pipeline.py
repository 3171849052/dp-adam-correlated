import csv
import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pytest
from exp1e import sweep


def summary(job):
    return dict(status='completed', smoke=False, method=job['method'], seed=job['seed'],
                lr=job['lr'], max_grad_norm=job.get('max_grad_norm'), eps_scale=job.get('eps_scale'),
                optimizer_steps=250, planned_total_steps=250, initialization_sha256=str(job['seed']),
                final=dict(test_top1=.3 if job['lr']==1e-3 else .2, train_loss=float(job['seed']%10)))


@pytest.mark.parametrize('code', [0,1])
def test_full_pipeline_queue_reuse_and_failures(tmp_path, monkeypatch, code):
    config = sweep.ROOT/'exp1e/config.yaml'
    monkeypatch.setattr(sweep,'ROOT',tmp_path)
    monkeypatch.setattr(sweep.torch.cuda,'device_count',lambda:4)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(sweep.time,'sleep',lambda _:None)
    live=set(); launched=[]; peak=0
    class Process:
        def __init__(self,cmd,cwd,env,stdout,stderr):
            nonlocal peak
            gpu=env['CUDA_VISIBLE_DEVICES']; assert gpu not in live
            live.add(gpu); peak=max(peak,len(live)); assert len(live)<=4
            self.gpu=gpu; self.polls=0
            job={key:convert(cmd[cmd.index(flag)+1]) for key,flag,convert in
                 [('method','--method',str),('seed','--seed',int),('lr','--lr',float)]}
            for key,flag in [('max_grad_norm','--max-grad-norm'),('eps_scale','--eps-scale')]:
                if flag in cmd: job[key]=float(cmd[cmd.index(flag)+1])
            directory=Path(cmd[cmd.index('--result-dir')+1])
            assert stdout.name==str(directory/'train.log')
            assert env['TMPDIR']==str(tmp_path/'exp1e/cache/tmp')
            if 'stage2' in str(directory):
                assert (tmp_path/'exp1e/results/selected_clip.json').is_file()
                assert len([d for d in launched if 'stage1' in str(d)])==5 and not any('stage1' in x for x in live)
            if '/final/' in str(directory):
                assert (tmp_path/'exp1e/results/selected_configs.json').is_file()
                assert len([d for d in launched if '/search/' in str(d)])==8
            launched.append(directory)
            if code==0:
                for name in ('config.yaml','metrics.csv','final.pt'): (directory/name).touch()
                np.save(directory/'train_order.npy',np.arange(10))
                if job['method']==sweep.SCALE:
                    (directory/'norm_stats.csv').touch(); (directory/'scale_stats.csv').touch()
                (directory/'summary.json').write_text(json.dumps(summary(job)))
        def poll(self):
            self.polls+=1
            if self.polls==1: return None
            live.remove(self.gpu)
            return code
    monkeypatch.setattr(sweep.subprocess,'Popen',Process)
    args=Namespace(config=config,smoke=False)
    if code:
        with pytest.raises(RuntimeError,match='Trial failed'): sweep.main(args)
        assert len(launched)==5 and not (tmp_path/'exp1e/results/selected_clip.json').exists()
    else:
        assert sweep.main(args)==0
        assert len(launched)==20
        base=tmp_path/'exp1e/results'
        assert json.loads((base/'selected_clip.json').read_text())['max_grad_norm']==100
        assert json.loads((base/'selected_configs.json').read_text())[sweep.SCALE]['lr']==1e-3
        second=list(csv.DictReader((base/'search/stage2_summary.csv').open()))
        assert len(second)==4 and sum('/stage1/' in r['result_dir'] for r in second)==1
        assert len(list(csv.DictReader((base/'final_multiseed.csv').open())))==12
        final=json.loads((base/'final_summary.json').read_text())
        assert all(r['seeds']==list(sweep.FINAL_SEEDS) and r['train_loss_std']==1 for r in final)
        sentinel=launched[0]/'preserve'; sentinel.write_text('keep')
        assert sweep.main(args)==0 and len(launched)==20 and sentinel.read_text()=='keep'
    assert peak==4 and not live


def test_incomplete_or_smoke_result_never_reused(tmp_path):
    (tmp_path/'summary.json').write_text(json.dumps(dict(status='completed',smoke=True)))
    with pytest.raises(AssertionError): sweep.read_completed(tmp_path,sweep.stage1_trials()[0])


def test_selection_ties():
    results=[(j,Path(j['name']),summary(j)) for j in reversed(sweep.stage1_trials())]
    assert sweep.best(results,'max_grad_norm')[0]['max_grad_norm']==100
    jobs=sweep.stage2_trials(100)
    results=[(j,Path(j['name']),dict(final=dict(test_top1=.5))) for j in reversed(jobs)]
    assert sweep.best(results,'lr')[0]['lr']==5e-4
