"""Read-only historical trials; verify original seals without importing old packages."""
from functools import lru_cache
from exp16c import ROOT,RESULTS
from exp16c.config import Trial,HISTORICAL_METHODS,STPEX,file_hash,json_hash,save_json
from exp16c.support import read

@lru_cache(None)
def verify_path(path):
    row=read(path/'summary.json');cfg=read(path/'config.json');seal=read(path/'completion.json')
    if row['status']!='completed':raise ValueError(f'Historical failure: {path}')
    h=cfg.pop('config_sha256')
    if json_hash(cfg)!=h or row['config_sha256']!=h:raise ValueError(f'Historical configuration changed: {path}')
    if seal['trial_id']!=row['trial_id'] or seal['core_sha256']!=row['code_sha256']:raise ValueError('Historical seal identity changed')
    for filename,digest in seal['file_sha256'].items():
        if file_hash(path/filename)!=digest:raise ValueError(f'Historical artifact changed: {path/filename}')
    for filename,digest in row['code_sha256'].items():
        if file_hash(ROOT/filename)!=digest:raise ValueError(f'Historical code changed: {filename}')
    for filename,digest in cfg['input_assets_sha256'].items():
        # CV historical manifests also include NLP assets; no need to open those.
        if 'cifar' in filename or 'cv_split' in filename or 'vit_tiny' in filename:
            if file_hash(ROOT/filename)!=digest:raise ValueError('Historical input changed')
    c=Trial(**{k:row[k] for k in Trial.__dataclass_fields__})
    expected=250 if c.stage=='final' else 225
    if row['optimizer_steps']!=expected or row['completed_epochs']!=5 or abs(row['actual_epsilon']-8)>1e-7:raise ValueError('Invalid historical protocol')
    if cfg['logical_batch_size']!=1000 or cfg['physical_batch_size']!=250 or cfg['precision']!='FP32':raise ValueError('Historical batch/precision mismatch')
    if row['official_evaluation_used']!=(c.stage=='final'):raise ValueError('Historical evaluation leak')
    if c.stage=='final' and cfg['official_assets_sha256']['data/cifar-100-python/test']!=file_hash(ROOT/'data/cifar-100-python/test'):raise ValueError('Historical official data changed')
    from exp16c.matrices import load
    matrix=load(c.workload,c.mechanism,c.total_steps)
    if row['mechanism']['matrix_sha256']!=file_hash(matrix['path']):raise ValueError('Historical matrix mismatch')
    return dict(row,source=str(path.relative_to(ROOT)),historical=True)

def rows(stage):
    out=[]
    for exp,methods in [('exp16a',HISTORICAL_METHODS),('exp16b',(STPEX,))]:
        if exp=='exp16b' and stage!='search':continue
        for path in sorted((ROOT/exp/'results'/stage/'cv').glob('*/summary.json')):
            row=read(path)
            if row['method'] in methods and row['status']=='completed':out.append(verify_path(path.parent))
    return out

def formal(cfg):
    path=ROOT/'exp16a/results/final/cv'/cfg.id
    if not path.exists():return None
    row=verify_path(path)
    if any(row[k]!=v for k,v in cfg.values().items()):raise ValueError('Historical formal configuration mismatch')
    return row

def winners():return {m:v for m,v in read(ROOT/'exp16a/results/final_freeze.json')['winners']['cv'].items() if m in HISTORICAL_METHODS}
