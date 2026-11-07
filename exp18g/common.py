"""Frozen Exp18g protocol and local assets."""
import csv
import hashlib
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from exp18g import BASE, ROOT

MODEL_PATH = ROOT/'cache/bert-tiny'
SEEDS = list(range(20261111,20261116))
T, K, EPOCHS, BATCH = 310, 62, 5, 1000
METHOD = 'adam-mf-sgd-stp-b2lt'
PARAMS = dict(lr=.01,clip=500,eps_scale=.001,beta1=.9,beta2=.999,adam_eps=1e-8,weight_decay=0)
PYTHON = '/home/longt29/miniconda3/envs/curve/bin/python'

def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path): return json.loads(path.read_text())
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def configs(stage='formal'):return read(BASE/f'plans/{stage}.json')
def outdir(cfg):
    return BASE/cfg['stage']/(f"gpu{cfg['gpu']}" if cfg['stage']=='smoke' else f"seed{cfg['seed']}")
def load_matrix():
    import numpy as np
    p=ROOT/'exp18d/matrices/sgd.npz'
    meta=read(p.with_suffix('.json'))
    with np.load(p) as f:return dict({k:f[k].copy() for k in f.files},metadata=meta)
def calibration(cfg):
    from exp18g.privacy import fixed_epoch_sensitivity,target_mu,epsilon_from_mu
    m=load_matrix();s=fixed_epoch_sensitivity(m['C'],5,62);mu=target_mu(8,1e-5)
    sigma=cfg['clip']*s/mu;epsilon=epsilon_from_mu(cfg['clip']*s/sigma,1e-5)
    assert abs(epsilon-8)<1e-9
    return dict(passed=True,seed=cfg['seed'],epsilon=epsilon,delta=1e-5,mu=mu,
        sensitivity=s,clip=cfg['clip'],sigma_sum=sigma,sigma_average=sigma/BATCH,
        T=T,participations=5,participation_interval=62,sampling_amplification=False,
        adjacency='add/remove zeroed fixed slot',clip_space='previous Adam vhat STP scaled space',
        workload='sgd',matrix_sha256=m['metadata']['sha256'],
        scope='One full training only; five-training composition is NOT epsilon=8')
def env(gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=read(BASE/'scheduling.json')['gpu_uuids'][str(gpu)],
        EXP18G_GPU=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    return e
def verify_seal():
    for filename in ('input_manifest.json','code_manifest.json'):
        root=ROOT if filename=='input_manifest.json' else BASE
        for rel,h in read(BASE/filename).items():assert sha(root/rel)==h,f'Changed input: {rel}'
def seal_code():
    paths=list(BASE.glob('*.py'))+list((BASE/'plans').glob('*.json'))
    paths += [BASE/n for n in ('protocol.json','hyperparameters.json','scheduling.json','input_manifest.json')]
    save(BASE/'code_manifest.json',{str(p.relative_to(BASE)):sha(p) for p in sorted(paths)})
def completed(cfg):
    out=outdir(cfg)
    if not (out/'complete.json').exists():return None
    for name,h in read(out/'complete.json').items():assert sha(out/name)==h
    r=read(out/'summary.json');assert r['status']=='complete' and r['config']==cfg and r['steps']==cfg['steps']
    trace=[json.loads(s) for s in (out/'steps.jsonl').read_text().splitlines()]
    assert [s['step'] for s in trace]==list(range(1,cfg['steps']+1))
    return r
