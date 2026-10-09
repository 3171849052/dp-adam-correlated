import copy
import csv
import json
import math
import numpy as np
import pytest
from exp8b.config import *
from exp8b import search,config,frozen,final,report,audit,launcher


def test_accuracy_only_and_t_statistics():
    a=dict(trial(IID,1e-4,1),status='completed',category='search',completed_epochs=5,accuracy=.71,validation_loss=.1)
    b=dict(a,lr=3e-4,accuracy=.72,validation_loss=99)
    assert search.best([a,b,dict(b,status='non_finite'),dict(b,category='smoke',accuracy=1.)])==b
    st=report.statistics(np.linspace(.7,.79,10))
    assert st['mean']==pytest.approx(.745)
    assert st['sample_std']==pytest.approx(np.linspace(.7,.79,10).std(ddof=1))
    assert (st['ci95'][1]-st['mean'])/st['standard_error']==pytest.approx(2.2621571628)


@pytest.mark.parametrize('method',[IID,BIAS_SCALE])
def test_search_budget_seed_and_boundary_once(tmp_path,monkeypatch,method):
    monkeypatch.setattr(config,'BASE',tmp_path);monkeypatch.setattr(config,'RESULTS',tmp_path/'results')
    monkeypatch.setattr(search,'RESULTS',tmp_path/'results');monkeypatch.setattr(search,'ROOT',tmp_path)
    eps=.1 if method.endswith('-scale') else None
    smoke_dir=trial_dir(trial(method,1e-4,1.,eps),'smoke');smoke_dir.mkdir(parents=True)
    (smoke_dir/'summary.json').write_text(json.dumps(dict(first_step_diagnostics=dict(gradient_norm_p50=50.))))
    started=[]
    def runner(jobs,gpu,category):
        result=[]
        for j in jobs:
            assert j['seed']==SEED and category=='search' and gpu==0
            started.append(j)
            directory=tmp_path/'results/search/trials'/trial_id(j);directory.mkdir(parents=True)
            with (directory/'mechanism_metrics.csv').open('w') as f:
                f.write('gradient_norm_p50,clip_fraction\n50,.8\n')
            # Deliberate edge winner exercises all allowed outward refinements.
            accuracy=.6+.01*math.log10(j['lr']/1e-5)+.005*math.log10(j['C'])
            result.append(dict(j,status='completed',category='search',completed_epochs=5,accuracy=accuracy,
                               result_dir=str(directory.relative_to(tmp_path))))
        return result
    winner=search.search_method(method,runner=runner)
    assert len(started)<= (18 if eps else 12) and len({trial_id(j) for j in started})==len(started)
    proof=json.loads((tmp_path/'results/search'/f'{method}_search.json').read_text())
    assert proof['trials']==len(started) and proof['winner']==winner
    assert sum(s['stage']=='refine_lr_once' for s in proof['stages'])==1
    assert len({j['lr'] for j in started if j['lr']>max(LRS)})<=1


def test_search_failure_excluded():
    r=dict(trial(IID,1e-4,1),status='non_finite',category='search',completed_epochs=0,accuracy=None)
    with pytest.raises(AssertionError): search.best([r])


def test_freeze_tamper_and_70_jobs(tmp_path,monkeypatch):
    results=tmp_path/'results';results.mkdir()
    monkeypatch.setattr(config,'BASE',tmp_path);monkeypatch.setattr(frozen,'RESULTS',results)
    monkeypatch.setattr(frozen,'ROOT',tmp_path)
    # Independent unit fixture for freeze semantics, with mocked audited training evidence.
    selected={m:dict(trial(m,1e-4,10.,.1 if m.endswith('-scale') else None),status='completed',category='search',
                         completed_epochs=5,official_validation_used=False,result_dir=f'trials/{i}',checkpoint_sha256='') for i,m in enumerate(METHODS)}
    for m,r in selected.items():
        d=tmp_path/r['result_dir'];d.mkdir(parents=True)
        (d/'checkpoint.pt').write_bytes(b'finite-fixture');r['checkpoint_sha256']=file_hash(d/'checkpoint.pt')
        (d/'summary.json').write_text(json.dumps(r))
    monkeypatch.setattr(frozen,'audit_trial',lambda p:next(r for r in selected.values() if p==tmp_path/r['result_dir']))
    for n in ('split.npz','tokens_128.pt'): (results/n).write_bytes(b'asset')
    model_path=tmp_path/'model';model_path.mkdir();(model_path/'pytorch_model.bin').write_bytes(b'pretrained')
    monkeypatch.setattr(frozen,'MODEL_PATH',model_path)
    save_json(results/'search/selected_configs.json',selected)
    frozen.freeze(selected);assert frozen.load_frozen()==selected
    jobs=final.final_jobs();assert len(jobs)==70 and len({trial_id(j) for j in jobs})==70
    assert {(j['method'],j['seed']) for j in jobs}=={(m,s) for m in METHODS for s in FINAL_SEEDS}
    (results/'frozen_configs.json').write_text((results/'frozen_configs.json').read_text()+' ')
    with pytest.raises(AssertionError): frozen.load_frozen()


def test_pairing_mismatch():
    keys=('initialization_sha256','classifier_initialization_sha256','pretrained_sha256','train_order_sha256','dropout_trace_sha256',
          'split_sha256','tokens_sha256','tokenizer_sha256','model_config_sha256')
    r=dict(seed=SEED,**{k:'same' for k in keys})
    audit.audit_pairing([r,dict(r)])
    with pytest.raises(AssertionError): audit.audit_pairing([r,dict(r,dropout_trace_sha256='different')])


def test_report_seventy_rows_and_pair_effects(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'BASE',tmp_path);monkeypatch.setattr(report,'RESULTS',tmp_path)
    selected={m:trial(m,1e-4,10.,.1 if m.endswith('-scale') else None) for m in METHODS}
    monkeypatch.setattr(frozen,'load_frozen',lambda:selected)
    records={}
    for seed in FINAL_SEEDS:
        for i,m in enumerate(METHODS):
            d=tmp_path/'final/trials'/f'{seed}_{i}';d.mkdir(parents=True);(d/'summary.json').write_text('{}')
            records[d]=dict(selected[m],seed=seed,accuracy=.7+i*.01+(seed-FINAL_SEEDS[0])*.001,
                            result_dir=str(d),**{k:'paired' for k in ('initialization_sha256','classifier_initialization_sha256','pretrained_sha256','train_order_sha256',
                            'dropout_trace_sha256','split_sha256','tokens_sha256','tokenizer_sha256','model_config_sha256')})
    monkeypatch.setattr(report,'audit_trial',lambda d:records[d])
    for n in ('frozen_configs.json','frozen_manifest.json'): (tmp_path/n).write_text('{}')
    report.final_report()
    with (tmp_path/'final_multiseed.csv').open() as f: assert len(list(csv.DictReader(f)))==70
    with (tmp_path/'paired_effects.csv').open() as f: effects=list(csv.DictReader(f))
    assert len(effects)==8 and all(int(e['wins'])==10 for e in effects)
    assert 'Three workloads' in (tmp_path/'final_report.md').read_text()


def test_stage2_runs_gate_tests_search_freeze_final_in_order(tmp_path,monkeypatch):
    import sys
    from exp8b import stage1,stage2
    monkeypatch.setattr(stage2,'RESULTS',tmp_path)
    monkeypatch.setattr(config,'BASE',tmp_path)
    monkeypatch.setattr(sys,'argv',['exp8b.stage2','--gpu','0'])
    events=[]
    monkeypatch.setattr(stage2,'check_platform',lambda:events.append('gate'))
    monkeypatch.setattr(stage1,'unit_tests',lambda prefix:events.append(('tests',prefix)))
    def searched(gpu):
        events.append(('search',gpu));(tmp_path/'frozen_configs.json').write_text('{}')
    monkeypatch.setattr(search,'search',searched)
    monkeypatch.setattr(final,'final',lambda gpu:events.append(('final',gpu)))
    stage2.main()
    assert events==['gate',('tests','stage2_'),('search',0),('final',0)]
    assert json.loads((tmp_path/'stage2_status.json').read_text())['final_trials']==70


def test_single_gpu_fifo_worker_and_reuse(tmp_path,monkeypatch):
    results=tmp_path/'results'
    monkeypatch.setattr(config,'BASE',tmp_path);monkeypatch.setattr(config,'RESULTS',results)
    monkeypatch.setattr(launcher,'ROOT',tmp_path);monkeypatch.setattr(launcher,'RESULTS',results)
    monkeypatch.setattr(audit,'RESULTS',results)
    monkeypatch.setattr(audit,'audit_trial',lambda d:json.loads((d/'summary.json').read_text()))
    jobs=[trial(IID,lr,1.) for lr in LRS[:2]]
    calls=[]
    def worker(cmd,cwd,env,stdout,stderr):
        j=jobs[len(calls)];calls.append(cmd)
        assert env['CUDA_VISIBLE_DEVICES']=='0' and cmd[cmd.index('--gpu')+1]=='0'
        save_json(trial_dir(j,'search')/'summary.json',dict(j,status='completed'))
        return 0
    monkeypatch.setattr(launcher.subprocess,'call',worker)
    assert len(launcher.run_queue(jobs,gpu=0,category='search'))==2
    assert audit.audit_fifo()==dict(status='passed',launches=2,gpus=[0])
    assert len(launcher.run_queue(jobs,gpu=0,category='search'))==2 and len(calls)==2
    events=[json.loads(x) for x in (results/'scheduler.jsonl').read_text().splitlines()]
    assert [x['event'] for x in events]==['start','finish','start','finish']


def test_multigpu_queue_exactly_two_slots_each(monkeypatch):
    import threading
    import time
    active={};peaks={};lock=threading.Lock()
    def fake_runner(jobs,gpu,category):
        with lock:
            active[gpu]=active.get(gpu,0)+1;peaks[gpu]=max(peaks.get(gpu,0),active[gpu])
        time.sleep(.05)
        with lock: active[gpu]-=1
        return [dict(jobs[0],gpu=gpu)]
    monkeypatch.setattr(launcher,'run_queue',fake_runner)
    q=launcher.MultiGPUQueue([0,1,2],2)
    try:
        jobs=[trial(IID,lr,1.) for lr in np.linspace(1e-5,1e-3,18)]
        r=q.run(jobs)
        assert [x['lr'] for x in r]==[x['lr'] for x in jobs]
        assert peaks=={0:2,1:2,2:2} and not any(active.values())
    finally: q.close()


def test_final_prepares_official_after_freeze_before_parallel_queue(monkeypatch):
    from exp8b import data
    events=[]
    jobs=[dict(i=i) for i in range(70)]
    monkeypatch.setattr(final,'check_platform',lambda:events.append('gate'))
    def frozen_jobs(): events.append('frozen');return jobs
    monkeypatch.setattr(final,'final_jobs',frozen_jobs)
    monkeypatch.setattr(data,'official_validation',lambda phase:events.append(('official',phase)))
    monkeypatch.setattr(final,'audit_pairing',lambda rows:events.append('pairing'))
    monkeypatch.setattr(report,'final_report',lambda:events.append('report'))
    class Queue:
        def run(self,values,gpu,category):
            assert values==jobs and events==['gate','frozen',('official','final')]
            events.append('queue')
            return [dict(status='completed') for _ in values]
    final.final(queue=Queue())
    assert events==['gate','frozen',('official','final'),'queue','pairing','report']
