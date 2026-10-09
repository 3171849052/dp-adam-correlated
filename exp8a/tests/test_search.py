"""Test choices and complete search routing without running Stage 2."""
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import pytest
from exp8a import RESULTS
from exp8a.config import Config
from exp8a.benchmark_batch import select_batch
from exp8a.benchmark_length import choose_length, train_lengths, benchmark_lengths
from exp8a.audit import audit_runs


def bench(length=64,batch=20,rate=2.,mem=5.,geometry='scale',status='completed'):
    return dict(max_length=length,physical_batch_size=batch,logical_steps_per_second=rate,
                peak_reserved_bytes=mem*2**30,headroom_fraction=1-mem/12,geometry=geometry,status=status)


def test_batch_harmonic_throughput_memory_tie_and_headroom():
    rows=[bench(batch=b,rate=rate,mem=mem,geometry=g) for b,rate,mem in ((20,2.,4.),(40,2.08,6.),(50,2.3,10.)) for g in ('standard','scale')]
    chosen,candidates=select_batch(rows)
    assert chosen['physical_batch_size']==20 and len(candidates)==2
    # A single fast mechanism cannot hide a slow Scale update.
    rows[-2:]=[bench(batch=50,rate=10.,mem=6.,geometry='standard'),bench(batch=50,rate=.1,mem=6.)]
    assert select_batch(rows)[0]['physical_batch_size']==20


def results(seeds=(1,2,3),gap=.002):
    return [dict(seed=s,max_length=n,accuracy=.8-(gap if n==32 else 0),geometry='scale') for s in seeds for n in (32,64,128)]


def test_length_paired_mean_uncertainty_and_fallback():
    benchmarks=[bench(length=n,geometry=g) for n in (32,64,128) for g in ('standard','scale')]
    assert choose_length(results((1,)),benchmarks)[0]==64
    assert choose_length(results(),benchmarks)[0]==32
    assert choose_length(results(gap=.004),benchmarks)[0]==64
    rows=results()
    for r in rows:
        if r['max_length']==32: r['accuracy'] += {1:.004,2:0.,3:-.004}[r['seed']]
    assert choose_length(rows,benchmarks)[0]==64


def test_paired_audit_detects_mismatch(monkeypatch,tmp_path):
    import exp8a.audit as audit
    monkeypatch.setattr(audit,'RESULTS',tmp_path)
    rows=results((1,))
    for r in rows:
        r.update(initialization_sha256='init',classifier_initialization_sha256='head',checkpoint_sha256='checkpoint',
                 split_sha256='split',optimizer_steps=310,noise_draws=310,full_run=True,official_validation_used=False,
                 epsilon=8.,examples=5349,physical_batch_size=20,lr=.001,C=3.,eps_scale=.3)
    audit_runs(rows)
    rows[0]['initialization_sha256']='different'
    with pytest.raises(AssertionError): audit_runs(rows)


def test_full_run_limit_shared_hyperparameters_and_pairing(monkeypatch,tmp_path):
    import exp8a.benchmark_length as module
    monkeypatch.setattr(module,'RESULTS',tmp_path)
    (tmp_path/'token_lengths.json').write_text(json.dumps(dict(truncation_ratio={'32':.055,'64':.00006,'128':0.},search_validation_label_counts=[2365,2984])))
    calls=[]
    def fake(cfg,gpu):
        calls.append(cfg)
        return dict(seed=cfg.seed,max_length=cfg.max_length,accuracy=.8,loss=.4)
    monkeypatch.setattr(module,'run',fake)
    cfg=Config(physical_batch_size=20,lr=.001,C=3.,eps_scale=.3)
    benchmarks=[bench(length=n,geometry=g) for n in (32,64,128) for g in ('standard','scale')]
    _,rows,_=train_lengths(cfg,0,benchmarks)
    assert len(rows)==len(calls)==9
    assert {(c.lr,c.C,c.eps_scale,c.physical_batch_size) for c in calls}=={(.001,3.,.3,20)}
    assert len({(c.seed,c.max_length) for c in calls})==9


def test_stage2_entry_and_no_automatic_execution():
    output=subprocess.run([sys.executable,'-B','-m','exp8a.stage2','--help'],capture_output=True,text=True,check=True)
    assert '--gpu' in output.stdout and '--pilots' in output.stdout
    from exp8a import stage1
    assert 'stage2.execute' not in Path(stage1.__file__).read_text()


def test_pilot_summary_fields_and_completed_pilot_reuse(monkeypatch,tmp_path):
    import exp8a.stage2 as stage2
    monkeypatch.setattr(stage2,'RESULTS',tmp_path)
    (tmp_path/'token_lengths.json').write_text(json.dumps(dict(search_validation_label_counts=[2365,2984])))
    calls=[]
    def fake(cfg,gpu,steps,output):
        calls.append(cfg)
        result=dict(status='completed',optimizer_steps=62,noise_draws=62,seed=cfg.seed,
                    physical_batch_size=cfg.physical_batch_size,max_length=cfg.max_length,
                    lr=cfg.lr,C=cfg.C,eps_scale=cfg.eps_scale,accuracy=.6+cfg.lr,loss=.6)
        output.mkdir(parents=True,exist_ok=True)
        (output/'summary.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(stage2,'run',fake)
    cfg=Config(physical_batch_size=250)
    # A previously completed first pilot counts toward the four-pilot limit.
    fake(cfg,0,62,tmp_path/'pilots/pilot1')
    chosen=stage2.pilots(cfg,0,4)
    assert len(calls)==4 and chosen.lr==.0005
    again=stage2.pilots(cfg,0,4)
    assert len(calls)==4 and again==chosen


def test_completed_batch_continuation_checks_protocol(monkeypatch,tmp_path):
    import exp8a.stage2 as stage2
    from exp8a.benchmark_batch import write_csv
    monkeypatch.setattr(stage2,'RESULTS',tmp_path)
    (tmp_path/'benchmarks').mkdir()
    rows=[]
    for batch in (8,20,40,50,100):
        for geometry in ('standard','scale'):
            row=bench(batch=batch,geometry=geometry)
            row.update(warmup_steps=1,measured_steps=3,seed=20261011,lr=.0005,C=1.,eps_scale=.1,oom=False)
            (tmp_path/'benchmarks'/f'{geometry}_length64_batch{batch}.json').write_text(json.dumps(row))
            rows.append(row)
    write_csv(tmp_path/'physical_batch_benchmark.csv',rows)
    chosen,loaded,candidates=stage2.completed_batch_search(Config())
    assert chosen['physical_batch_size']==8 and len(loaded)==10
    with pytest.raises(AssertionError): stage2.completed_batch_search(Config(C=3.))


def test_equal_majority_baseline_is_insufficient_length_evidence():
    baseline=2984/5349
    rows=results()
    for row in rows: row['accuracy']=baseline
    benchmarks=[bench(length=n,geometry=g) for n in (32,64,128) for g in ('standard','scale')]
    length,reason=choose_length(rows,benchmarks,baseline_accuracy=baseline)
    assert length==64 and reason['convergence_evidence'] is False


def test_final_batch_workers_preserve_initial_benchmark_artifacts(monkeypatch,tmp_path):
    import exp8a.benchmark_batch as module
    monkeypatch.setattr(module,'RESULTS',tmp_path)
    calls=[]
    def fake(cfg,geometry,gpu,category):
        calls.append(category)
        return dict(bench(batch=cfg.physical_batch_size,geometry=geometry),oom=False)
    monkeypatch.setattr(module,'isolated',fake)
    module.search(path=tmp_path/'physical_batch_benchmark.csv')
    assert set(calls)=={'benchmarks'}
    calls.clear()
    module.search(path=tmp_path/'physical_batch_final_benchmark.csv')
    assert set(calls)=={'physical_batch_final_benchmark'}


def test_pilot_baseline_prevents_new_full_runs(monkeypatch,tmp_path):
    import exp8a.stage2 as stage2
    monkeypatch.setattr(stage2,'RESULTS',tmp_path)
    (tmp_path/'token_lengths.json').write_text(json.dumps(dict(search_validation_label_counts=[2365,2984])))
    calls=[]
    def fake(cfg,gpu,steps,output):
        calls.append(cfg)
        return dict(status='completed',optimizer_steps=62,noise_draws=62,seed=cfg.seed,
                    physical_batch_size=cfg.physical_batch_size,max_length=64,
                    lr=cfg.lr,C=cfg.C,eps_scale=cfg.eps_scale,accuracy=2984/5349,loss=.717)
    monkeypatch.setattr(stage2,'run',fake)
    with pytest.raises(AssertionError,match='usable convergence'):
        stage2.pilots(Config(),0,4)
    assert len(calls)==4 and not json.loads((tmp_path/'pilots.json').read_text())['converged']
