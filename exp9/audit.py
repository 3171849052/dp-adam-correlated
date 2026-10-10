"""Artifact integrity, per-run GDP claims, paired RNGs and scheduler audit."""
import json
from pathlib import Path
from exp9 import BASE, ROOT, RESULTS
from exp9.config import Trial, file_hash, save_json

def code_hashes():
    paths = sorted(BASE.rglob('*.py')) + [BASE/'kernel_provenance.json']
    return {str(p.relative_to(BASE)):file_hash(p) for p in paths if 'runtime' not in p.parts}

def protected_snapshot():
    """Record size/mtime of all existing experiment, raw-data and cache files."""
    roots = [p for p in ROOT.glob('exp*') if p.is_dir() and p != BASE] + [ROOT/'data',ROOT/'cache']
    return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
            for root in roots for p in root.rglob('*') if p.is_file()}

def audit_trial(cfg, allow_failure=False):
    summary = json.loads((cfg.output/'summary.json').read_text())
    assert all(summary[k]==v for k,v in cfg.values().items()) and summary['trial_id']==cfg.id
    if summary['status'] != 'completed':
        assert allow_failure and summary['status']=='non_finite', f'Inspect failed trial: {cfg.output}'
        return summary
    assert summary['code_sha256']==code_hashes(), f'Source changed: {cfg.output}'
    assert summary['assets_manifest_sha256']==file_hash(RESULTS/'assets_manifest.json')
    for name, expected in summary['file_sha256'].items(): assert file_hash(cfg.output/name)==expected, name
    resolved = json.loads((cfg.output/'config.json').read_text())
    assert resolved['total_steps']==cfg.total_steps and resolved['physical_batch_size']==cfg.physical_batch
    assert resolved['privacy']==cfg.protocol()['privacy']
    assert summary['finite'] and summary['noise_draws']==summary['optimizer_steps']==(2 if cfg.stage=='smoke' else cfg.total_steps)
    assert summary['completed_epochs']==(0 if cfg.stage=='smoke' else 5)
    assert summary['physical_batches']==summary['optimizer_steps']*(1000//cfg.physical_batch)
    assert abs(summary['calibration']['verified_full_epsilon']-cfg.epsilon)<1e-7
    assert summary['actual_epsilon']<=cfg.epsilon+1e-7
    assert summary['official_evaluation_used']==(cfg.stage in ('final','sweep'))
    if cfg.stage!='smoke':
        assert abs(summary['actual_epsilon']-cfg.epsilon)<1e-7
        assert summary['epochs'][-1]['validation_examples']==((10000 if cfg.task=='cv' else 872) if cfg.stage in ('final','sweep') else (5000 if cfg.task=='cv' else 5349))
    if cfg.stage in ('final','sweep'):
        assert resolved['frozen_manifest_sha256']==file_hash(RESULTS/'frozen_manifest.json')
        assert resolved['official_assets_sha256']==file_hash(RESULTS/'official_assets.json')
    return summary

def audit_pairing(rows):
    groups={}
    for row in rows:
        if row['status']!='completed': continue
        cfg=Trial(**{k:row[k] for k in Trial.__dataclass_fields__})
        key=(row['task'],row['seed'],cfg.total_steps,row['stage'])
        hashes=tuple(row[k] for k in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace'))
        if key in groups: assert groups[key]==hashes, f'Paired RNG mismatch: {key}'
        else: groups[key]=hashes
    return dict(status='passed',groups=len(groups))

def audit_fifo(path=None):
    path = path or RESULTS/'scheduler.jsonl'
    events=[json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []
    active={}; counts={}; last_index={}; peak={}
    for event in events:
        if event['event']=='start':
            queue=event['queue'];index=event['queue_index']
            assert index > last_index.get(queue,-1);last_index[queue]=index
            gpu=event['gpu'];task=event['task']
            existing=[v for v in active.values() if v['gpu']==gpu]
            assert all(v['task']==task for v in existing), 'Mixed CV/NLP GPU'
            assert len(existing)<event['capacity']
            peak.setdefault(gpu,{'cv':0,'nlp':0})
            peak[gpu][task]=max(peak[gpu][task],len(existing)+1)
            active[event['trial_id']]=event;counts[gpu]=counts.get(gpu,0)+1
        elif event['event'] in ('finish','interrupted'): active.pop(event['trial_id'])
    assert not active, 'Unfinished worker in scheduler log'
    return dict(status='passed',gpu_starts=counts,peak_concurrency=peak,events=len(events))

def verify_stage1():
    platform=json.loads((RESULTS/'platform_validation.json').read_text())
    assert platform['status']=='passed' and platform['full_experiments_started'] is False
    assert platform['code_sha256']==code_hashes(), 'Rerun Stage 1 after source changes'
    for name, expected in platform['evidence_sha256'].items(): assert file_hash(RESULTS/name)==expected
    assets=json.loads((RESULTS/'assets_manifest.json').read_text())
    for task in ('cv','nlp'): assert file_hash(RESULTS/f'{task}_split.npz')==assets['splits'][task]['file_sha256']
    assert file_hash(RESULTS/'nlp_tokens_128.pt')==assets['tokens_sha256']
    from exp9.cv_model import checkpoint_path
    assert file_hash(checkpoint_path())==assets['cv_pretrained']
    for name, expected in assets['nlp_pretrained'].items(): assert file_hash(ROOT/'cache/bert-tiny'/name)==expected
    assert file_hash(ROOT/'data/cifar-100-python/train')==assets['cifar_train']
    assert file_hash(ROOT/'data/sst2/train.parquet')==assets['sst2_train']
    smoke=json.loads((RESULTS/'smoke_summary.json').read_text())
    for row in smoke: audit_trial(Trial(**{k:row[k] for k in Trial.__dataclass_fields__}))
    audit_pairing(smoke);audit_fifo()
    return platform

def full_audit():
    rows=[];failures=[]
    for path in sorted(RESULTS.glob('*/*/*/summary.json')):
        row=json.loads(path.read_text());cfg=Trial(**{k:row[k] for k in Trial.__dataclass_fields__})
        if row['status']=='completed': rows.append(audit_trial(cfg))
        else: failures.append(dict(trial_id=cfg.id,status=row['status'],error=row.get('error')))
    result=dict(status='passed',completed=len(rows),failures=failures,pairing=audit_pairing(rows),fifo=audit_fifo(),
                privacy_scope='Each completed training only; no epsilon=8 claim for the entire selection pipeline.',
                evaluation_limitations=['CV historical experiments used official test for selection.',
                                        'NLP official validation has already been viewed by historical experiments; no new blind-test claim.'])
    save_json(RESULTS/'audit_summary.json',result);return result

if __name__=='__main__': print(json.dumps(full_audit(),indent=2))
