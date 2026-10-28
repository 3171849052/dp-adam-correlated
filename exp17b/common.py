"""Local artifact identity, atomic writes, and the frozen experiment protocol."""
import hashlib
import json
import os
from pathlib import Path
from exp17b import BASE, ROOT, RESULTS

PYTHON = '/home/longt29/miniconda3/envs/curve/bin/python'
METHODS = ('smooth','lc','gh')
LR = 0.001187378144539193
R = 2.8312235256706146
PROTOCOL = dict(dataset='CIFAR-100',model='pretrained ViT-Tiny',dtype='float32',
    clip=30., beta1=.9,beta2=.999,eps=1e-8,weight_decay=0.,delta=1e-5,
    epsilon=8.,expected_batch=1000,physical_batch=250,lr=LR,r=R,
    search=dict(population=45000,steps=225),final=dict(population=50000,steps=250),
    alphas=[0,.25,.5,1],gpus=[2,3],
    B=dict(seeds=list(range(20261101,20261104)),steps=50,configurations=7),
    C=dict(seeds=list(range(20261104,20261109)),steps=225,configurations=3),
    C2=dict(seeds=list(range(20261104,20261107)),steps=225,r=[0,1],configurations=4),
    D=dict(seeds=list(range(20261111,20261121)),steps=250,configurations=[3,5]),
    selection='B: per method rank by mean validation CE ascending, mean accuracy descending, alpha ascending. '
              'C2: choose positive same-r Smooth-minus-GH mean CE improvement exceeding main-r improvement; '
              'compare all r on the same three C2 seeds. Largest improvement wins, tie lower r.',
    mc=dict(seed=17002,histories=10,samples=100000,t=[1,2,10,50,250],
            snr=[0,.25,.5,1,2,4],r=[0,1,R],paths=['constant','alternating','varying','sparse']),
    privacy='Each run separately accounted by Opacus RDP. Short runs retain 225-step sigma. '
            'No joint epsilon=8 claim for tuning and repetitions. Validation assumed public; '
            'raw diagnostics and checkpoints are private audit artifacts.',
    interpretation='Fixed-state Monte Carlo does not establish adaptive training utility. '
                   'Curvature comparisons must match r, learning rate and seed. Official test has historical exposure.',
    resume='Atomic full-state checkpoints every 10 steps; source/assets/config and artifact hashes required; no silent retry.')


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
    from exp17b.cv_model import checkpoint_path
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
