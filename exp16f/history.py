"""Read-only, artifact-verified historical formal comparisons."""
from functools import lru_cache
from exp16f import ROOT,RESULTS
from exp16f.config import Trial,file_hash,json_hash,save_json,FINAL_SEEDS
from exp16f.support import read

def config(r):return Trial(**{k:r[k] for k in Trial.__dataclass_fields__})
def key(r):return tuple((r.values() if isinstance(r,Trial) else r)[k] for k in Trial.__dataclass_fields__)
@lru_cache(None)
def digest(p):return file_hash(p)
def verify(path):
    r=read(path/'summary.json');c=read(path/'config.json');seal=read(path/'completion.json')
    assert r['status']=='completed' and r['official_evaluation_used'] and r['seed'] in FINAL_SEEDS
    assert r['optimizer_steps']==r['noise_draws']==(250 if r['task']=='cv' else 310)
    assert r['completed_epochs']==5 and abs(r['actual_epsilon']-8)<1e-7
    assert r['epochs'][-1]['validation_examples']==(10000 if r['task']=='cv' else 872)
    h=c.pop('config_sha256');assert json_hash(c)==h==r['config_sha256']
    assert seal['trial_id']==r['trial_id']==json_hash({k:r[k] for k in Trial.__dataclass_fields__})[:20]
    assert seal['core_sha256']==r['code_sha256']==c['code_sha256']
    for mapping,base in [(seal['file_sha256'],path),(r['file_sha256'],path),(c['code_sha256'],ROOT),(c['input_assets_sha256'],ROOT),(c['official_assets_sha256'],ROOT)]:
        for name,h in mapping.items():assert digest(base/name)==h,(path,name)
    for k in Trial.__dataclass_fields__:assert c[k]==r[k]
    assert c['precision']=='FP32' and c['full_finetuning'] and c['epochs']==5
    assert c['logical_batch_size']==1000 and c['physical_batch_size']==(250 if r['task']=='cv' else 1000)
    assert c['privacy']['adjacency']=='add_remove_zero_out' and not c['privacy']['sampling_amplification']
    assert digest(ROOT/r['mechanism']['source'])==r['mechanism']['matrix_sha256']
    return dict(r,source=str(path.relative_to(ROOT)),historical=True,summary_sha256=digest(path/'summary.json'))
def collect():
    audit=read(ROOT/'exp16e/results/final_audit.json');assert audit['status']=='completed'
    for name,h in audit['output_sha256'].items():assert digest(ROOT/'exp16e/results'/name)==h
    import csv
    result=[]
    for row in csv.DictReader((ROOT/'exp16e/results/formal_per_seed.csv').open()):
        r=verify(ROOT/row['source']);assert r['method']==row['method'] and r['task']==row['task'] and str(r['seed'])==row['seed']
        result.append(r)
    for task in ('cv','nlp'):
        baseline=[r for r in result if r['task']==task and r['method']=='adam-mf-sgd-stp-b2lt']
        assert sorted(r['seed'] for r in baseline)==list(FINAL_SEEDS)
    save_json(RESULTS/'historical_formal.json',result)
    save_json(RESULTS/'history_audit.json',dict(status='passed',count=len(result),baseline_count=20,sources=sorted(set(r['source'].split('/')[0] for r in result))))
    return result
