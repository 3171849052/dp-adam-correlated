"""Read-only historical assets, provenance, and local output helpers."""
import csv
import json
import subprocess
from functools import lru_cache
import numpy as np
from exp15c import ROOT,BASE,RESULTS
from exp15c.config import file_hash,array_hash


def read(path): return json.loads(path.read_text())


def code_hashes():
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(BASE.glob('*.py'))}


def version(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def protected_snapshot():
    return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
        for name in ('exp9','exp14','exp15','exp15b') for p in (ROOT/name).rglob('*') if p.is_file()}


@lru_cache(None)
def historical_matrices(cfg):
    stage='final' if cfg.formal else 'search'
    for path in sorted((ROOT/'exp9/results'/stage/cfg.task).glob('*/summary.json')):
        r=read(path)
        if r['method']!='dp-adam-momentum-bias-bandinvmf' or r['status']!='completed': continue
        matrix=path.parent/'matrices.npz'
        if file_hash(matrix)!=r['file_sha256']['matrices.npz']: raise ValueError('Historical matrix checksum')
        with np.load(matrix) as f: d,S,W=(f[k].copy() for k in ('coefficients','strategy','workload'))
        assert S.shape==(cfg.total_steps,cfg.total_steps)
        return d,S,W,str(matrix.relative_to(ROOT))
    raise FileNotFoundError('Historical MF missing')


def matrices(cfg):
    from exp15c.bandinvmf import native_matrices
    d,S,W=native_matrices(cfg.total_steps)
    return d,S,W,.1*W,dict(source=f'exp15c/results/mf/momentum_{cfg.total_steps}.npz',
        individually_optimized_for_momentum=True,coefficients_sha256=array_hash(d),strategy_sha256=array_hash(S),
        filter_design_workload_sha256=array_hash(W),actual_workload_sha256=array_hash(.1*W))


def verify_assets():
    assets=read(ROOT/'exp9/results/assets_manifest.json')
    from exp15c.cv_model import checkpoint_path
    paths={ROOT/'data/cifar-100-python/train':assets['cifar_train'],ROOT/'data/sst2/train.parquet':assets['sst2_train'],
           ROOT/'exp9/results/nlp_tokens_128.pt':assets['tokens_sha256'],checkpoint_path():assets['cv_pretrained']}
    for name,h in assets['nlp_pretrained'].items(): paths[ROOT/'cache/bert-tiny'/name]=h
    for task in ('cv','nlp'): paths[ROOT/f'exp9/results/{task}_split.npz']=assets['splits'][task]['file_sha256']
    for path,h in paths.items():
        if file_hash(path)!=h: raise ValueError(f'Asset changed: {path}')
    if not (ROOT/'exp9/results/nlp_official_tokens_128.pt').is_file(): raise FileNotFoundError('Offline official tokens missing')
    return {str(p.relative_to(ROOT)):h for p,h in paths.items()}


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def result_dir(row):
    if row.get('reuse'): return ROOT/row['reuse']['source_directory']
    from exp15c.config import Trial
    return Trial(**{k:row[k] for k in Trial.__dataclass_fields__}).output
