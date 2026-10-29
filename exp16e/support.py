"""Provenance and output utilities; no historical package imports."""
import csv,json,subprocess
from functools import lru_cache
from exp16e import ROOT,BASE,RESULTS
from exp16e.config import file_hash,save_json
CORE=('__init__','config','train','data','cv_model','nlp_model','optimizer','clipping','privacy','noise','matrices','support')
def read(p):return json.loads(p.read_text())
def code_hashes():return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(BASE.glob('*.py'))}
def platform_hashes():return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(BASE.glob('*.py'))}
def version():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def write_csv(path,rows):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def protected_snapshot():
    return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns] for p in ROOT.rglob('*')
        if p.is_file() and not p.is_relative_to(BASE) and '.git' not in p.parts}
def verify_protected():
    if protected_snapshot()!=read(RESULTS/'protected_before.json'):raise ValueError('Read-only files changed')
@lru_cache(None)
def verify_assets():
    assets=read(ROOT/'exp9/results/assets_manifest.json')
    from exp16e.cv_model import checkpoint_path
    paths={ROOT/'data/cifar-100-python/train':assets['cifar_train'],ROOT/'data/sst2/train.parquet':assets['sst2_train'],
        ROOT/'exp9/results/nlp_tokens_128.pt':assets['tokens_sha256'],checkpoint_path():assets['cv_pretrained']}
    for name,h in assets['nlp_pretrained'].items():paths[ROOT/'cache/bert-tiny'/name]=h
    for task in ('cv','nlp'):paths[ROOT/f'exp9/results/{task}_split.npz']=assets['splits'][task]['file_sha256']
    for p,h in paths.items():
        if file_hash(p)!=h:raise ValueError(f'Asset mismatch: {p}')
    return {**{str(p.relative_to(ROOT)):h for p,h in paths.items()},**read(RESULTS/'local_assets.json')['files']}

def official_assets():
    # Called only after 28 configurations have been frozen, before official workers.
    frozen=read(RESULTS/'final_freeze.json')
    if len(frozen['winners'])!=2:raise ValueError('Final configuration freeze missing')
    files=('data/cifar-100-python/test','data/sst2/validation.parquet','exp9/results/nlp_official_tokens_128.pt')
    hashes={f:file_hash(ROOT/f) for f in files}
    historical=read(ROOT/'exp9/results/official_assets.json')
    expected=dict(zip(files,(historical['cifar_test_sha256'],historical['sst2_validation_sha256'],historical['tokens_sha256'])))
    if hashes!=expected:raise ValueError('Official asset mismatch')
    hashes.update(read(RESULTS/'official_local_assets.json')['files'])
    return hashes
