import json
from exp14c import RESULTS
from exp14c.config import Trial, file_hash, save_json
from exp14c.support import code_hashes, protected_snapshot, historical, reference_dir

def audit_row(r):
    c=Trial(**{k:r[k] for k in Trial.__dataclass_fields__});directory=c.output
    assert r['trial_id']==c.id and r['status']=='completed' and r['finite']
    assert r['code_sha256']==code_hashes()
    for name,h in r['file_sha256'].items(): assert file_hash(directory/name)==h
    assert r['noise_draws']==r['optimizer_steps']==(2 if c.stage=='smoke' else c.total_steps)
    assert abs(r['calibration']['verified_full_epsilon']-8)<1e-7
    assert r['official_evaluation_used']==(c.stage=='final')
    if c.stage!='smoke':
        assert len(r['epochs'])==5 and abs(r['actual_epsilon']-8)<1e-7
        assert r['epochs'][-1]['validation_examples']==((10000 if c.task=='cv' else 872) if c.formal else (5000 if c.task=='cv' else 5349))
    if c.stage=='final':
        config=json.loads((directory/'config.json').read_text())
        assert config['frozen_configs_sha256']==file_hash(RESULTS/'frozen_manifest.json')
        ref=json.loads((reference_dir(c)/'summary.json').read_text())
        for k in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace','strategy_sha256','workload_sha256','coefficients'):
            assert r[k]==ref[k],(c.id,k)
    diagnostics=[json.loads(line) for line in (directory/'optimizer_metrics.jsonl').read_text().splitlines()]
    assert diagnostics and all(x['nonfinite_parameter_tensors']==0 for x in diagnostics)
    return r

def main():
    search=json.loads((RESULTS/'search_results.json').read_text());formal=json.loads((RESULTS/'formal_results.json').read_text())
    assert len(search)==120 and len(formal)==80
    for r in search+formal: audit_row(r)
    manifest=json.loads((RESULTS/'frozen_manifest.json').read_text())
    assert file_hash(RESULTS/'frozen_manifest.json')==json.loads((RESULTS/'frozen_manifest.sha256.json').read_text())['sha256']
    for name,h in manifest['search_evidence_sha256'].items(): assert file_hash(RESULTS/name)==h
    from exp14c.pipeline import rank
    for w in manifest['winners']:
        assert w['trial_id']==rank([r for r in search if r['task']==w['task'] and r['method']==w['method']])[0]['trial_id']
    active={}
    for line in (RESULTS/'scheduler.jsonl').read_text().splitlines():
        e=json.loads(line)
        if e['event']=='launched': assert e['gpu'] not in active;active[e['gpu']]=e['trial_id']
        if e['event']=='finished': assert active.pop(e['gpu'])==e['trial_id']
    assert not active
    assert protected_snapshot()==json.loads((RESULTS/'protected_before.json').read_text())
    save_json(RESULTS/'audit.json',dict(status='passed',search=120,formal=80,strict_exp9_pairing=True,history_unchanged=True,one_worker_per_gpu=True,freeze_evidence_verified=True))
if __name__=='__main__': main()
