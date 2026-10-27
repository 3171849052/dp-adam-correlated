"""Local artifact identity, atomic writes, and the frozen experiment protocol."""
import hashlib
import json
import os
from pathlib import Path
from exp17 import BASE, ROOT, RESULTS

PYTHON = '/home/longt29/miniconda3/envs/curve/bin/python'
METHODS = ('adam','bc','lc','hc')
PROTOCOL = dict(dataset='CIFAR-100',model='pretrained ViT-Tiny',dtype='float32',
    clip=30., beta1=.9,beta2=.999,eps=1e-8,weight_decay=0.,delta=1e-5,
    epsilon=8.,expected_batch=1000,physical_batch=250,
    search=dict(population=45000,steps=225),final=dict(population=50000,steps=250),
    lr_range=[1e-5,3e-3],r_ranges=dict(bc=[1e-4,1.],lc=[1e-3,4.],hc=[1e-3,4.]),
    search_seeds=[20261101,20261102,20261103],review_seeds=[20261104,20261105],
    final_seeds=list(range(20261111,20261121)),gpus=[1,2,3],
    fine_rule='For each dimension use n log-spaced interior points between the midpoints '
              'of the winning coarse log-coordinate and its adjacent coarse coordinates. '
              'At boundaries truncate at the global limit. n=4 for lr, n=3 for r. '
              'Exclude the coarse winning r when symmetric: replace it by the lower '
              'three-eighths interval point. Fine lr never equals coarse lr; all fine pairs are new.',
    selection='Rank config means by accuracy descending, cross entropy ascending, config hash ascending. '
              'Review top two with two NEW seeds; rerank using ALL actual repetitions.',
    ablation='Two extra full final runs per paired seed: LC and HC with correction disabled, '
             'same frozen lr,r and seeds; 20 extra runs excluded from 248 main runs.',
    privacy='Opacus RDP per run only. Validation is assumed public; tuning, repeated runs, '
            'raw training diagnostics and checkpoints have no joint epsilon=8 release guarantee.',
    theory='LC is conditional on fixed preceding states. HC fourth-order bias reasoning applies '
           'to fixed clipped-gradient paths with fixed positive smoothing, not adaptive training. '
           'Scaling lambda2 with nu2 is not a uniform fourth-order guarantee near zero.',
    schedule='Constant learning rate, five passes in expectation; evaluation at each 45/50 steps. '
             'Independent Bernoulli membership each step, fixed normalization 1000; empty draws update.',
    reuse='Only matching frozen source/assets/config and SHA256-verified successful artifacts. '
          'Interrupted trials resume from last atomic full-state step checkpoint; failures require explicit retry.')


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
    from exp17.cv_model import checkpoint_path
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
