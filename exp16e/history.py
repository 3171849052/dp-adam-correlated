"""Read historical seals, never import historical packages or use controlled runs."""
from functools import lru_cache
from pathlib import Path
from exp16e import ROOT,RESULTS
from exp16e.config import Trial,METHODS,SEARCH_SEED,FINAL_SEEDS,file_hash,json_hash
from exp16e.support import read

@lru_cache(None)
def digest(path):return file_hash(path)
def config(r):return Trial(**{k:r[k] for k in Trial.__dataclass_fields__})
def key(r):
    v=r.values() if isinstance(r,Trial) else r
    return tuple(v[k] for k in Trial.__dataclass_fields__)

def verify(path):
    row=read(path/'summary.json');cfg=read(path/'config.json');seal=read(path/'completion.json');c=config(row)
    assert row['status']=='completed' and row['finite']
    assert seal['trial_id']==row['trial_id']==c.id
    h=cfg.pop('config_sha256');assert json_hash(cfg)==h==row['config_sha256']
    assert seal['core_sha256']==row['code_sha256']==cfg['code_sha256']
    for name,h in seal['file_sha256'].items():assert digest(path/name)==h,f'Historical artifact changed: {path/name}'
    for name,h in row['file_sha256'].items():assert digest(path/name)==h
    for name,h in row['code_sha256'].items():assert digest(ROOT/name)==h,f'Historical source changed: {name}'
    for name,h in cfg['input_assets_sha256'].items():assert digest(ROOT/name)==h
    for k,v in c.values().items():assert cfg[k]==v
    expected=c.protocol()
    for k in ('epochs','total_steps','logical_batch_size','physical_batch_size','privacy','precision','full_finetuning','evaluation','participation','noise_seed','dropout_seed'):
        assert cfg[k]==expected[k],(path,k,cfg[k],expected[k])
    if c.method.startswith('momentum'):
        assert cfg['optimizer']['name']=='Momentum'
        assert (ROOT/'exp16c/momentum_optimizer.py').read_bytes()==(ROOT/'exp15c/optimizer.py').read_bytes()
    assert cfg['optimizer']==expected['optimizer'],(path,cfg['optimizer'],expected['optimizer'])
    if c.scaled:assert cfg['stp'].replace('exp16b original','exp15 original')==expected['stp']
    assert row['optimizer_steps']==row['noise_draws']==c.total_steps and row['completed_epochs']==5
    assert row['physical_batches']==c.total_steps*1000//c.physical_batch
    assert abs(row['actual_epsilon']-8)<1e-7 and row['official_evaluation_used']==(c.stage=='final')
    assert row['epochs'][-1]['validation_examples']==(10000 if c.task=='cv' else 872) if c.stage=='final' else row['epochs'][-1]['validation_examples']==(5000 if c.task=='cv' else 5349)
    if c.stage=='final':
        assert c.seed in FINAL_SEEDS
        for name,h in cfg['official_assets_sha256'].items():assert digest(ROOT/name)==h
    from exp16e.matrices import load
    matrix=load(c.workload,c.mechanism,c.total_steps)
    assert row['mechanism']['matrix_sha256']==digest(matrix['path'])
    return dict(row,source=str(path.relative_to(ROOT)),historical=True,summary_sha256=digest(path/'summary.json'),completion_sha256=digest(path/'completion.json'))

def incumbents():
    winners={'cv':{},'nlp':{}}
    for exp in ('exp16a','exp16c','exp16d'):
        for task,methods in read(ROOT/exp/'results/final_freeze.json')['winners'].items():
            winners[task].update({m:v for m,v in methods.items() if m in METHODS})
    return winners

def collect(stage):
    result={}
    for exp in ('exp16a','exp16b','exp16c','exp16d'):
        allowed_file=ROOT/exp/'results'/f'{stage}_results.json'
        allowed={r['trial_id'] for r in read(allowed_file)} if allowed_file.exists() else None
        # Only original selection results, excluding controlled-only review runs.
        for p in sorted((ROOT/exp/'results'/stage).glob('*/*/summary.json')):
            r=read(p)
            if r['method'] not in METHODS or r['status']!='completed':continue
            if allowed is not None and r['trial_id'] not in allowed:continue
            v=verify(p.parent);result[key(v)]=v
    return list(result.values())

def formal(c):
    for exp in ('exp16d','exp16c','exp16a'):
        directory=ROOT/exp/'results/final'/c.task
        # Equality deliberately normalizes numeric int/float representation, but no mechanism aliases.
        for p in directory.glob('*/summary.json'):
            r=read(p)
            if key(r)==key(c):return verify(p.parent)
    return None
