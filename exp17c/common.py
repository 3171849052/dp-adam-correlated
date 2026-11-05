"""Local artifact identity, atomic writes, and the frozen experiment protocol."""
import hashlib
import json
import os
from pathlib import Path
from exp17c import BASE, ROOT, RESULTS

PYTHON = '/home/longt29/miniconda3/envs/curve/bin/python'
METHODS = ('smooth','lc','gh')
LR = 0.001187378144539193
R = 0.
PROTOCOL = dict(dataset='CIFAR-100',model='pretrained ViT-Tiny',dtype='float32',
 clip=30.,beta1=.9,beta2=.999,eps=1e-8,delta=1e-5,epsilon=8.,
 expected_batch=1000,physical_batch=250,lr=LR,r=0.,
 alphas=[0,1e-14,3e-14,1e-13,3e-13,1e-12,3e-12,1e-11,3e-11,1e-10],gpus=[1,2,3],
 B=dict(seeds=list(range(1703101,1703104)),steps=50),
 C=dict(seeds=list(range(1703201,1703206)),steps=225),
 D=dict(seeds=list(range(1703301,1703311)),steps=250),
 selection='Final validation CE mean ascending, accuracy mean descending, alpha ascending.',
 boundary='If best positive alpha is a boundary and beats baseline, add one adjacent decade (three seeds, 50 steps); at most one extension before C.',
 privacy='Per-run RDP only; repeated tuning does not have joint epsilon=8 guarantee.',
 resume='Full state every 10 steps; rerun start_background after interruption.')


def save(path, obj):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temp,path)


def read(path): return json.loads(Path(path).read_text())
def digest(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def code_hashes():
    return {str(p.relative_to(BASE)):sha(p) for p in sorted(BASE.rglob('*.py')) if 'runtime' not in p.parts}

def assets(formal=False):
    from exp17c.cv_model import checkpoint_path
    paths=[checkpoint_path(),ROOT/'data/cifar-100-python/train',ROOT/'data/cifar-100-python/meta',ROOT/'exp9/results/cv_split.npz']
    if formal: paths.append(ROOT/'data/cifar-100-python/test')
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def trial_id(cfg): return cfg['stage']+'_'+cfg['method']+'_'+digest(cfg)[:16]
def output(cfg): return RESULTS/'trials'/trial_id(cfg)

def success(cfg, identity):
    folder=output(cfg)
    if not (folder/'summary.json').exists(): return None
    s=read(folder/'summary.json')
    if s['status']!='completed': return None
    if s['config']!=cfg or s['identity']!=identity: raise RuntimeError('Successful result identity mismatch: '+str(folder))
    for name,value in s['hashes'].items():
        if sha(folder/name)!=value: raise RuntimeError('Artifact changed: '+str(folder/name))
    return s
