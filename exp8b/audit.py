"""Recompute mathematical/accounting evidence and verify stored artifacts."""
from dataclasses import fields
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from exp8b import BASE, ROOT, RESULTS
from exp8b.config import *
from exp8b.train import mechanism,array_hash
from exp8b.privacy import epsilon_from_mu


def source_hashes():
    return {str(p.relative_to(BASE)):file_hash(p) for p in sorted(BASE.rglob('*.py')) if 'runtime' not in p.parts}


def audit_trial(directory):
    directory=Path(directory)
    assert directory.resolve().is_relative_to(BASE)
    r=json.loads((directory/'summary.json').read_text())
    assert r['status']=='completed'
    c=json.loads((directory/'config.json').read_text())
    cfg=Config(**{f.name:c[f.name] for f in fields(Config)})
    assert c==dict(cfg.protocol(),category=r['category'],frozen_configs_sha256=r['frozen_configs_sha256'])
    assert all(r[k]==getattr(cfg,k) for k in ('method','seed','lr','C','eps_scale'))
    smoke=r['category']=='smoke'; steps=1 if smoke else 310
    assert (r['optimizer_steps'],r['noise_draws'],r['physical_batches'],r['physical_batch_size'],r['max_length'])==(steps,steps,steps,1000,128)
    assert r['planned_total_steps']==310 and r['finite'] and r['completed_epochs']==(0 if smoke else 5)
    assert r['official_validation_used']==(r['category']=='final')
    assert r['validation_examples']==(0 if smoke else 872 if r['category']=='final' else 5349)
    metrics=list(csv.DictReader((directory/'metrics.csv').open()))
    assert [int(x['step']) for x in metrics]==([1] if smoke else [62,124,186,248,310])
    assert r['accuracy']==float(metrics[-1]['accuracy'])
    for x,y in zip(metrics,r['epochs']):
        assert all(float(x[k])==y[k] for k in y)
        assert all(np.isfinite(float(v)) for v in x.values())
    diagnostics=list(csv.DictReader((directory/'mechanism_metrics.csv').open()))
    assert len(diagnostics)==steps and [int(x['step']) for x in diagnostics]==list(range(1,steps+1))
    assert all(np.isfinite(float(v)) for x in diagnostics for v in x.values())
    d,S,W,meta,p=mechanism(cfg)
    with np.load(directory/'matrices.npz') as m:
        for key,value in (('coefficients',d),('strategy',S),('workload',W)):
            np.testing.assert_allclose(m[key],value,rtol=1e-12,atol=1e-12)
    assert r['coefficients']==d.tolist() and r['strategy_sha256']==array_hash(S) and r['workload_sha256']==array_hash(W)
    assert r['calibration']==p and r['matrix_optimization']==meta
    assert np.isclose(epsilon_from_mu(cfg.C*p['sensitivity']/p['innovation_std_sum'],1e-5),8,atol=1e-8)
    if not smoke: assert np.isclose(float(metrics[-1]['epsilon']),8,atol=1e-8)
    for k,name in (('matrix_sha256','matrices.npz'),('checkpoint_sha256','checkpoint.pt'),('train_order_sha256','train_order.npy'),
                   ('config_sha256','config.json'),('metrics_sha256','metrics.csv'),('mechanism_metrics_sha256','mechanism_metrics.csv')):
        assert r[k]==file_hash(directory/name)
    for k,path in (('pretrained_sha256',MODEL_PATH/'pytorch_model.bin'),('split_sha256',RESULTS/'split.npz'),
                   ('tokens_sha256',RESULTS/'tokens_128.pt'),('tokenizer_sha256',MODEL_PATH/'vocab.txt'),('model_config_sha256',MODEL_PATH/'config.json')):
        assert r[k]==file_hash(path)
    assert r['pretrained_sha256']==CHECKPOINT_SHA256
    with np.load(RESULTS/'split.npz') as split: np.testing.assert_array_equal(np.load(directory/'train_order.npy'),split['train'])
    trace=[cfg.seed+100000+s for s in range(steps)]
    assert r['dropout_trace_sha256']==hashlib.sha256(json.dumps(trace).encode()).hexdigest()
    checkpoint=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert checkpoint['logical_steps']==steps and all(torch.isfinite(v).all() for v in checkpoint['model'].values())
    assert len(checkpoint['optimizer']['param_groups'])==1
    g=checkpoint['optimizer']['param_groups'][0]
    assert (g['lr'],tuple(g['betas']),g['eps'],g['weight_decay'])==(cfg.lr,(.9,.999),1e-8,0.)
    assert len(checkpoint['optimizer']['state'])==len(g['params'])
    for state in checkpoint['optimizer']['state'].values():
        assert int(state['step'])==steps and all(torch.isfinite(state[k]).all() for k in ('exp_avg','exp_avg_sq'))
    if r['category']=='final':
        from exp8b.frozen import load_frozen
        frozen=load_frozen()
        assert all(r[k]==frozen[cfg.method][k] for k in ('lr','C','eps_scale'))
        assert r['frozen_configs_sha256']==file_hash(RESULTS/'frozen_configs.json')
    return r


def audit_pairing(records):
    for seed in {r['seed'] for r in records}:
        group=[r for r in records if r['seed']==seed]
        for key in ('initialization_sha256','classifier_initialization_sha256','pretrained_sha256','train_order_sha256',
                    'dropout_trace_sha256','split_sha256','tokens_sha256','tokenizer_sha256','model_config_sha256'):
            assert len({r[key] for r in group})==1,(seed,key)


def audit_fifo():
    path=RESULTS/'scheduler.jsonl'
    live={}; launches=0; gpus=set()
    for line in path.read_text().splitlines() if path.exists() else []:
        event=json.loads(line);gpus.add(event['gpu'])
        if event['event']=='start':
            key=(event['trial_id'],event['category'])
            assert key not in live
            assert sum(g==event['gpu'] for g in live.values())<2, 'More than two concurrent trials on a GPU'
            live[key]=event['gpu']; launches+=1
        else:
            assert event['event']=='finish'
            assert live.pop((event['trial_id'],event['category']))==event['gpu']
    assert not live and gpus.issubset({0,1,2})
    return dict(status='passed',launches=launches,gpus=sorted(gpus))


def check_platform():
    p=json.loads((RESULTS/'platform_validation.json').read_text())
    assert p['status']=='passed' and p['unit_tests']=='passed' and p['smoke_methods']==list(METHODS)
    assert p['source_hashes']==source_hashes(), 'Source changed: rerun Stage 1'
    assert p['unit_tests_sha256']==file_hash(RESULTS/'unit_tests.log')
    assert p['tests_xml_sha256']==file_hash(RESULTS/'tests.xml')
    results=[audit_trial(trial_dir(trial(m,1e-4,1.,.1 if METHODS[m]['geometry']=='scale' else None),'smoke')) for m in METHODS]
    assert p['smoke_summary_hashes']=={r['method']:file_hash(ROOT/r['result_dir']/'summary.json') for r in results}
    audit_pairing(results)
    audit_fifo()
    return p


def audit_previous_experiments():
    snapshot=json.loads((RESULTS/'previous_experiments.json').read_text())
    for name,meta in snapshot.items():
        p=ROOT/name
        assert p.exists() and [p.stat().st_size,p.stat().st_mtime_ns]==meta,name
    return dict(status='passed',files=len(snapshot),check='size and nanosecond modification time; prior experiments untouched')

if __name__=='__main__':
    check_platform()
    records=[audit_trial(p.parent) for p in RESULTS.glob('*/trials/*/summary.json') if json.loads(p.read_text())['status']=='completed']
    save_json(RESULTS/'artifact_audit.json',dict(audited_trials=len(records),fifo=audit_fifo(),previous_experiments=audit_previous_experiments()))
