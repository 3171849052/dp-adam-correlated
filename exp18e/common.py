"""Exp18e plans, local artifacts, isolation and frozen input hashes."""
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from exp18e import BASE, ROOT

PYTHON = '/home/longt29/miniconda3/envs/curve/bin/python'
MODEL_PATH = ROOT / 'cache/bert-tiny'
SEEDS = [20261111, 20261112, 20261113, 20261114, 20261115]
T, K, EPOCHS, BATCH = 310, 62, 5, 1000
GROUPS = {'adam':['adam-iid','adam-b2lt-sgd'],
          'momentum':['momentum-iid','momentum-b2lt-momentum'],
          'stpex2':['stpex2-iid','stpex2-b2lt-momentum-bias']}
WORKLOADS = {'adam':'sgd','momentum':'momentum','stpex2':'momentum-bias'}
FIXED = {'adam-iid':dict(lr=.002,clip=1,eps=1e-8),
         'adam-b2lt-sgd':dict(lr=.01,clip=40,eps=1e-8),
         'momentum-iid':dict(lr=2.,clip=1,eps=1e-8),
         'momentum-b2lt-momentum':dict(lr=.3,clip=20,eps=1e-8)}
HISTORICAL = dict(lr=.08,clip=100,eps=.005)
SEARCH_STAGES = ('coarse','fine','boundary','review')


def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path): return json.loads(path.read_text())
def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(tmp,path)
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)
def candidate_id(params): return digest(params)[:16]
def make_config(stage,method,seed,params):
    group = next(g for g,names in GROUPS.items() if method in names)
    return dict(stage=stage,method=method,group=group,seed=seed,
                steps=3 if stage=='smoke' else T,**params)
def configs(stage='formal'):
    if stage in SEARCH_STAGES: return read(BASE/'plans'/f'{stage}.json')
    stp = HISTORICAL if stage=='smoke' else read(BASE/'selection_freeze.json')['parameters']
    return [make_config(stage,m,seed,FIXED[m] if m in FIXED else stp)
            for seed in (SEEDS[:1] if stage=='smoke' else SEEDS)
            for methods in GROUPS.values() for m in methods]
def outdir(cfg):
    name = f"{cfg['method']}_seed{cfg['seed']}"
    if cfg['stage'] in SEARCH_STAGES:
        params = {k:cfg[k] for k in ('lr','clip','eps')}
        return BASE/'search'/cfg['stage']/candidate_id(params)/name
    return BASE/cfg['stage']/name

def load_matrix(name):
    import numpy as np
    path = BASE/'matrices'/f'{name}.npz'; meta=read(path.with_suffix('.json'))
    assert sha(path)==meta['sha256']
    with np.load(path) as f: return dict({k:f[k].copy() for k in f.files},metadata=meta)
def calibration(cfg):
    # Recompute for EVERY trial, including every clipping candidate and smoke run.
    import numpy as np
    from exp18e.privacy import target_mu,fixed_epoch_sensitivity,epsilon_from_mu
    iid = cfg['method'].endswith('-iid')
    m = None if iid else load_matrix(WORKLOADS[cfg['group']])
    C = np.eye(T) if iid else m['C']
    sens = fixed_epoch_sensitivity(C,EPOCHS,K); mu=target_mu(8,1e-5)
    sigma = cfg['clip']*sens/mu
    epsilon = epsilon_from_mu(cfg['clip']*sens/sigma,1e-5)
    assert abs(epsilon-8)<1e-9
    return dict(epsilon=epsilon,delta=1e-5,mu=mu,sensitivity=sens,clip=cfg['clip'],
        sigma_sum=sigma,sigma_average=sigma/BATCH,participations=5,participation_interval=K,T=T,
        sampling_amplification=False,adjacency='add/remove zeroed fixed slot',
        clip_space='previous-d preconditioned' if cfg['group']=='stpex2' else 'original',
        matrix_sha256=None if iid else m['metadata']['sha256'],
        scope='one full training; not composition of search/formal trials')
def env(gpu):
    e=os.environ.copy()
    e.update(CUDA_VISIBLE_DEVICES=read(BASE/'scheduling.json')['gpu_uuids'][str(gpu)],
             EXP18E_GPU=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1')
    return e

def seal():
    paths=list(BASE.glob('*.py'))+list((BASE/'tests').glob('*.py'))
    paths += [BASE/n for n in ('protocol.json','search_space.json','scheduling.json','assets.json',
                               'upstream_manifest.json','start_background.sh')]
    paths += list((BASE/'matrices').glob('*'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
def verify_seal():
    accepted=read(BASE/'acceptance.json')
    assert accepted['passed'] and seal()==accepted['code'],'Accepted source/config changed'
    for rel,h in read(BASE/'upstream_manifest.json').items():
        assert sha(ROOT/rel)==h,f'Upstream changed: {rel}'
    for rel,h in read(BASE/'assets.json').items():
        assert sha(ROOT/rel)==h,f'Asset changed: {rel}'
def completed(cfg):
    out=outdir(cfg)
    if not (out/'complete.json').exists(): return None
    assert read(out/'config.json')==cfg
    for name,h in read(out/'complete.json').items(): assert sha(out/name)==h,f'Artifact changed: {out/name}'
    trace=[json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in trace]==list(range(1,cfg['steps']+1))
    summary=read(out/'summary.json')
    assert summary['status']=='complete' and summary['steps']==cfg['steps']
    return summary


def verify_freeze():
    for rel,h in read(BASE/'freeze_manifest.json').items():
        assert sha(BASE/rel)==h,f'Frozen selection changed: {rel}'
    frozen=read(BASE/'selection_freeze.json')
    assert frozen['frozen'] and not frozen['official_validation_used']
    hp=read(BASE/'hyperparameters.json')
    assert hp['selection_sha256']==sha(BASE/'selection_freeze.json')
    assert hp['methods']['stpex2-iid']==hp['methods']['stpex2-b2lt-momentum-bias']==frozen['parameters']
    assert configs()==read(BASE/'plans/formal.json') and len(configs())==30
