"""Source, protected-file, matrix, per-training privacy and pairing audits."""
import json
from pathlib import Path
import numpy as np
from exp12 import BASE, ROOT, RESULTS
from exp12.config import Trial, METHOD, FINAL_SEEDS, file_hash, array_hash, save_json


def code_hashes():
    paths = sorted(BASE.glob('*.py')) + sorted((BASE/'tests').glob('*.py')) + [BASE/'source_provenance.json']
    return {str(p.relative_to(BASE)):file_hash(p) for p in paths}


def protected_snapshot():
    roots = [ROOT/name for name in ('exp9','exp10','exp10b','exp11','data','cache')]
    return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
            for root in roots for p in root.rglob('*') if p.is_file()}


def audit_trial(cfg, allow_failure=False):
    r = json.loads((cfg.output/'summary.json').read_text())
    assert r['status']=='completed', f"{cfg.output}: {r.get('error')}"
    assert r['trial_id']==cfg.id and all(r[k]==v for k,v in cfg.values().items())
    assert r['code_sha256']==code_hashes()
    assert r['assets_manifest_sha256']==file_hash(RESULTS/'assets_manifest.json')
    for name, expected in r['file_sha256'].items():
        assert file_hash(cfg.output/name)==expected, name
    c = json.loads((cfg.output/'config.json').read_text())
    assert c['privacy']==cfg.protocol()['privacy'] and c['optimizer']==cfg.protocol()['optimizer']
    assert c['total_steps']==cfg.total_steps and c['physical_batch_size']==250
    assert r['finite'] and r['noise_draws']==r['optimizer_steps']==(2 if cfg.stage=='smoke' else cfg.total_steps)
    assert r['completed_epochs']==(0 if cfg.stage=='smoke' else 5)
    assert r['physical_batches']==4*r['optimizer_steps']
    assert abs(r['calibration']['verified_full_epsilon']-8)<1e-7
    assert r['actual_epsilon']<=8+1e-7
    assert r['official_evaluation_used']==(cfg.stage=='final')
    if cfg.stage!='smoke':
        assert abs(r['actual_epsilon']-8)<1e-7
        assert r['epochs'][-1]['validation_examples']==(10000 if cfg.stage=='final' else 5000)
    if cfg.stage=='final':
        assert c['frozen_manifest_sha256']==file_hash(RESULTS/'frozen_manifest.json')
        assert c['official_assets_sha256']==file_hash(RESULTS/'official_assets.json')
    from exp12.bandinvmf import bias_matrices
    d,S,W,_=bias_matrices(cfg.total_steps)
    with np.load(cfg.output/'matrices.npz') as stored:
        for key,value in [('coefficients',d),('strategy',S),('workload',W)]:
            np.testing.assert_array_equal(stored[key],value)
    assert r['strategy_sha256']==array_hash(S) and r['workload_sha256']==array_hash(W)
    return r


def baseline_audit():
    """Never infer pairing from the seed alone; verify historical artifacts first."""
    from exp12.privacy import calibrate, epsilon_from_mu
    historical=ROOT/'exp9/results'
    rows=json.loads((historical/'final_results.json').read_text())
    rows=sorted([r for r in rows if r['task']=='cv' and r['method']==METHOD],key=lambda r:r['seed'])
    assert [r['seed'] for r in rows]==list(FINAL_SEEDS)
    winner=json.loads((historical/'frozen_configs.json').read_text())['cv'][METHOD]
    sources={name:file_hash(ROOT/'exp9'/name) for name in ('bandinvmf.py','clipping.py','cv_model.py','noise.py','privacy.py','train.py','data.py','nlp_model.py','config.py')}
    assets=json.loads((RESULTS/'assets_manifest.json').read_text())
    old_assets=json.loads((historical/'assets_manifest.json').read_text())
    assert old_assets['cv_pretrained']==assets['cv_pretrained'] and old_assets['cifar_train']==assets['cifar_train']
    assert old_assets['splits']['cv']['indices_sha256']==assets['splits']['cv']['indices_sha256']
    with np.load(RESULTS/'matrices/T250/matrices.npz') as old:
        d,S,W=(old[key].copy() for key in ('old_coefficients','old_strategy','old_workload'))
    evidence=[]
    for r in rows:
        out=historical/r['result_dir']
        assert json.loads((out/'summary.json').read_text())==r
        assert r['status']=='completed' and r['stage']=='final' and r['epsilon']==8 and r['eps_scale'] is None
        assert all(r[k]==winner[k] for k in ('lr','C','eps_scale'))
        assert all(r['code_sha256'][k]==v for k,v in sources.items()), 'Historical source mismatch'
        assert r['completed_epochs']==5 and r['optimizer_steps']==r['noise_draws']==250
        assert r['physical_batches']==1000 and r['physical_batch_size']==250
        assert r['official_evaluation_used'] and r['epochs'][-1]['validation_examples']==10000
        assert r['accuracy']==r['epochs'][-1]['accuracy'] and abs(r['actual_epsilon']-8)<1e-7
        for name,h in r['file_sha256'].items():
            assert file_hash(out/name)==h, f'Historical artifact mismatch: {out/name}'
        cfg=json.loads((out/'config.json').read_text())
        reference=Trial('cv',METHOD,r['lr'],r['C'],seed=r['seed'],stage='final').protocol()
        for key in ('optimizer','privacy','dataset_size','total_steps','epochs','logical_batch_size','physical_batch_size','gradient_accumulation','full_finetuning','rng'):
            assert cfg[key]==reference[key], f'Historical protocol mismatch: {key}'
        assert cfg['assets_manifest_sha256']==file_hash(historical/'assets_manifest.json')
        assert cfg['frozen_manifest_sha256']==file_hash(historical/'frozen_manifest.json')
        assert cfg['official_assets_sha256']==file_hash(historical/'official_assets.json')
        with np.load(out/'matrices.npz') as matrix:
            for key,value in [('coefficients',d),('strategy',S),('workload',W)]:
                np.testing.assert_allclose(matrix[key],value,rtol=1e-13,atol=1e-13)
        p=calibrate(S,reference)
        assert np.isclose(p['innovation_std_sum'],r['calibration']['innovation_std_sum'],rtol=1e-12)
        assert abs(epsilon_from_mu(r['C']*p['sensitivity']/p['innovation_std_sum'],1e-5)-8)<1e-7
        evidence.append(dict(seed=r['seed'],trial_id=r['trial_id'],accuracy=r['accuracy'],
                    summary_sha256=file_hash(out/'summary.json'),verified_artifacts=r['file_sha256']))
    save_json(RESULTS/'baseline_audit.json',dict(status='passed',method=METHOD,lr=winner['lr'],C=winner['C'],
              source_sha256=sources,runs=evidence,paired_rng_status='pending Exp12 formal runs'))
    return rows


def pairing(new_rows,old_rows):
    old={r['seed']:r for r in old_rows}
    output=[]
    for r in sorted(new_rows,key=lambda r:r['seed']):
        baseline=old[r['seed']]
        keys=('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace')
        mismatches=[key for key in keys if r[key]!=baseline[key]]
        output.append(dict(seed=r['seed'],exp12_accuracy=r['accuracy'],exp9_accuracy=baseline['accuracy'],
                           pairable=not mismatches,mismatches=mismatches,
                           difference=r['accuracy']-baseline['accuracy'] if not mismatches else None))
    return output


def scheduler_audit():
    events=[json.loads(line) for line in (RESULTS/'scheduler.jsonl').read_text().splitlines()]
    active={};peak={};counts={}
    for e in events:
        if e['event']=='start':
            gpu=e['gpu'];assert gpu not in active, 'Multiple CV workers on one GPU'
            active[gpu]=e['trial_id'];peak[gpu]=1;counts[gpu]=counts.get(gpu,0)+1
        elif e['event'] in ('finish','interrupted'):
            assert active.pop(e['gpu'])==e['trial_id']
    assert not active
    return dict(status='passed',gpu_starts=counts,peak_workers_per_gpu=peak)
