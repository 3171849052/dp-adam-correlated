"""Exp18f fixed protocol, independent plans and per-training privacy calibration."""
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from exp18f import BASE, ROOT

PYTHON = sys.executable
MODEL_PATH = ROOT/'cache/bert-tiny'
TOKENS = ROOT/'exp18e/assets/nlp_tokens_128.pt'
SPLIT = ROOT/'exp18e/assets/nlp_split.npz'
METHODS = ('stpex2-iid','stpex2-b2lt-momentum-bias')
PARAMS = ('lr','clip','eps')
BASELINE = dict(lr=.03,clip=100,eps=.001)
T,K,EPOCHS,BATCH = 310,62,5,1000
SEARCH_SEEDS = [20261121,20261122]
TURN_SEEDS = [20261123,20261124,20261125]
REVIEW_SEEDS = [20261131,20261132,20261133,20261134,20261135]

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
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)
def make_config(stage,candidate,method,seed,params):
    return dict(stage=stage,candidate=candidate,method=method,group='stpex2',seed=seed,
                steps=3 if stage=='smoke' else T,**params)
def configs(stage):return read(BASE/'plans'/f'{stage}.json')
def outdir(cfg):
    return BASE/('smoke' if cfg['stage']=='smoke' else 'search')/cfg['stage']/cfg['candidate']/f"{cfg['method']}_seed{cfg['seed']}"
def load_matrix(name='momentum-bias'):
    import numpy as np
    path=BASE/'matrices'/f'{name}.npz';meta=read(path.with_suffix('.json'))
    assert sha(path)==meta['sha256']
    with np.load(path) as f:return dict({k:f[k].copy() for k in f.files},metadata=meta)
def calibration(cfg):
    import numpy as np
    from exp18f.privacy import target_mu,fixed_epoch_sensitivity,epsilon_from_mu
    iid=cfg['method'].endswith('-iid');m=None if iid else load_matrix()
    C=np.eye(T) if iid else m['C']
    sensitivity=fixed_epoch_sensitivity(C,EPOCHS,K);mu=target_mu(8,1e-5)
    sigma=cfg['clip']*sensitivity/mu
    epsilon=epsilon_from_mu(cfg['clip']*sensitivity/sigma,1e-5)
    assert abs(epsilon-8)<1e-9
    return dict(epsilon=epsilon,delta=1e-5,mu=mu,sensitivity=sensitivity,clip=cfg['clip'],
        sigma_sum=sigma,sigma_average=sigma/BATCH,participations=5,participation_interval=K,T=T,
        sampling_amplification=False,adjacency='add/remove zeroed fixed slot',
        clip_space='previous-d preconditioned',matrix_sha256=None if iid else m['metadata']['sha256'],
        scope='one full training only; search/review composition is NOT epsilon=8')
def env(gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=read(BASE/'scheduling.json')['gpu_uuids'][str(gpu)],
        EXP18F_GPU=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1')
    return e

def terminal(cfg,verify=False):
    out=outdir(cfg)
    if not (out/'terminal.json').exists():return None
    marker=read(out/'terminal.json');result=read(out/'summary.json')
    assert read(out/'config.json')==cfg
    assert result['status'] in ('complete','numerical_failure')
    if verify:
        for name,h in marker['files'].items():assert sha(out/name)==h,f'Changed artifact: {out/name}'
        trace=[json.loads(l) for l in (out/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in trace]==list(range(1,len(trace)+1))
        if result['status']=='complete':assert len(trace)==cfg['steps']==result['steps']
    return result

def seal():
    paths=list(BASE.glob('*.py'))+list((BASE/'tests').glob('*.py'))
    paths += [BASE/n for n in ('protocol.json','search_space.json','scheduling.json','inputs.json','start_background.sh','plans/axis.json','plans/smoke.json')]
    paths += list((BASE/'matrices').glob('*'))
    return {str(p.relative_to(BASE)):sha(p) for p in sorted(paths)}
def verify_seal():
    accepted=read(BASE/'acceptance.json')
    assert accepted['passed'] and seal()==accepted['code'],'Accepted code/config changed'
    for rel,h in read(BASE/'inputs.json').items():assert sha(ROOT/rel)==h,f'Input changed: {rel}'
