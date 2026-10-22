"""Provenance and immutable historical mechanism lookup."""
import csv
import json
import subprocess
from functools import lru_cache
import numpy as np
from exp15d import ROOT, BASE, RESULTS
from exp15d.config import file_hash, array_hash, save_json
from exp15d.bandinvmf import inverse_strategy, momentum_bias_workload


def code_hashes():
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(BASE.rglob('*.py')) if 'runtime' not in p.parts and 'results' not in p.parts}


def version():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def protected_snapshot():
    roots=[p for p in ROOT.glob('exp*') if p.is_dir() and p!=BASE]+[ROOT/'data',ROOT/'cache']
    return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
            for root in roots for p in root.rglob('*') if p.is_file()}


@lru_cache(None)
def reference_dir(task, formal):
    stage='final' if formal else 'search'
    for p in sorted((ROOT/'exp9/results'/stage/task).glob('*/summary.json')):
        r=json.loads(p.read_text())
        if r['method']=='dp-adam-momentum-bias-bandinvmf' and r['status']=='completed':
            return p.parent
    raise FileNotFoundError('Missing verified Exp9 Momentum-Bias mechanism')


def matrices(cfg):
    if cfg.method=='momentum+iid':
        n=cfg.total_steps
        d=np.array([1.],dtype=np.float64);S=np.eye(n,dtype=np.float64)
        t=np.arange(1,n+1);lag=t[:,None]-t[None,:]
        M=np.where(lag>=0,.9**np.maximum(lag,0),0.)
        actual=.1*np.tril(np.cumsum(M,axis=0));W=momentum_bias_workload(n,.9)
        return d,S,W,actual,dict(source='exp15d IID identity strategy',
            coefficients_sha256=array_hash(d),strategy_sha256=array_hash(S),
            filter_design_workload_sha256=array_hash(W),actual_workload_sha256=array_hash(actual),
            controlled_filter=False,individually_optimized_for_momentum=False,
            covariance='innovation_std_sum^2 * I_T',full_schedule_participations=5,
            sensitivity_squared=5.,independent_logical_steps=True)
    directory=reference_dir(cfg.task,cfg.formal)
    summary=json.loads((directory/'summary.json').read_text())
    path=directory/'matrices.npz'
    if file_hash(path)!=summary['file_sha256']['matrices.npz']: raise ValueError('Historical matrix file hash mismatch')
    with np.load(path) as saved: d,S,W=(saved[k].copy() for k in ('coefficients','strategy','workload'))
    if len(d)!=4 or S.shape!=(cfg.total_steps,cfg.total_steps): raise ValueError('Wrong mechanism horizon')
    np.testing.assert_allclose(S,inverse_strategy(d,cfg.total_steps),rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(W,momentum_bias_workload(cfg.total_steps,.9))
    if array_hash(S)!=summary['strategy_sha256'] or array_hash(W)!=summary['workload_sha256']:
        raise ValueError('Historical matrix content hash mismatch')
    t=np.arange(1,cfg.total_steps+1);lag=t[:,None]-t[None,:]
    M=np.where(lag>=0,.9**np.maximum(lag,0),0.)
    if cfg.method not in ('momentum','momentum+iid'): M/=(1-.9**t)[:,None]
    actual=.1*np.tril(np.cumsum(M,axis=0))
    return d,S,W,actual,dict(source=str(path.relative_to(ROOT)),source_sha256=file_hash(path),
        coefficients_sha256=array_hash(d),strategy_sha256=array_hash(S),
        filter_design_workload_sha256=array_hash(W),actual_workload_sha256=array_hash(actual),
        workload_factor_omitted_by_Exp9=.1,controlled_filter=True,
        individually_optimized_for_momentum=False)


def verify_assets():
    manifest=ROOT/'exp9/results/assets_manifest.json';assets=json.loads(manifest.read_text())
    from exp15d.cv_model import checkpoint_path
    paths={ROOT/'data/cifar-100-python/train':assets['cifar_train'],
           ROOT/'data/sst2/train.parquet':assets['sst2_train'],
           ROOT/'exp9/results/nlp_tokens_128.pt':assets['tokens_sha256'],
           checkpoint_path():assets['cv_pretrained']}
    for name,h in assets['nlp_pretrained'].items(): paths[ROOT/'cache/bert-tiny'/name]=h
    for task in ('cv','nlp'): paths[ROOT/f'exp9/results/{task}_split.npz']=assets['splits'][task]['file_sha256']
    for path,h in paths.items():
        if file_hash(path)!=h: raise ValueError(f'Asset hash mismatch: {path}')
    return dict(manifest_sha256=file_hash(manifest),files={str(p.relative_to(ROOT)):h for p,h in paths.items()})


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def result_dir(row):
    if row.get('reuse'):
        return ROOT/row['reuse']['source_directory']
    from exp15d.config import Trial
    return Trial(**{k:row[k] for k in Trial.__dataclass_fields__}).output
