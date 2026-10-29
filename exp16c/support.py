"""Provenance and output utilities; no historical package imports."""
import csv,json,subprocess
from functools import lru_cache
from exp16c import ROOT,BASE,RESULTS
from exp16c.config import file_hash,save_json
CORE=('__init__','config','train','data','cv_model','randomness','momentum_optimizer','stpex_optimizer','optimizer','clipping','privacy','noise','matrices','support')
def read(p):return json.loads(p.read_text())
def code_hashes():return {f'exp16c/{n}.py':file_hash(BASE/(n+'.py')) for n in CORE}
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
    from exp16c.cv_model import checkpoint_path
    paths={ROOT/'data/cifar-100-python/train':assets['cifar_train'],checkpoint_path():assets['cv_pretrained'],
           ROOT/'exp9/results/cv_split.npz':assets['splits']['cv']['file_sha256']}
    for p,h in paths.items():
        if file_hash(p)!=h:raise ValueError(f'Asset mismatch: {p}')
    return {str(p.relative_to(ROOT)):h for p,h in paths.items()}

def official_assets():
    frozen=read(RESULTS/'final_freeze.json')
    if set(frozen['winners']['cv'])!=set(__import__('exp16c.config',fromlist=['METHODS']).METHODS):raise ValueError('Incomplete freeze')
    path=ROOT/'data/cifar-100-python/test'
    h=file_hash(path)
    if h!=read(ROOT/'exp9/results/official_assets.json')['cifar_test_sha256']:raise ValueError('Official asset mismatch')
    return {str(path.relative_to(ROOT)):h}
