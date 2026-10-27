"""Fixed experimental protocol and canonical training identity."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,math
from exp16b import ROOT,BASE,RESULTS
MODEL_PATH=ROOT/'cache/bert-tiny'
SPLIT_SEED=20261008
SEARCH_SEED=20261101
METHODS=('stpex-mf-momentum-bias-bandinvmf','stpex-mf-momentum-bias-b2lt')
def json_hash(x): return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def array_hash(x):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def save_json(path,x):
    path=Path(path).resolve()
    if not path.is_relative_to(BASE):raise ValueError(f'Output outside exp16b: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temporary.replace(path)
@dataclass(frozen=True)
class Trial:
    task:str
    method:str
    lr:float
    clip:float|None=None
    eps_scale:float|None=None
    seed:int=SEARCH_SEED
    stage:str='search'
    smoke_phase:str='search'
    replicate:int=0
    def __post_init__(self):
        if self.seed!=SEARCH_SEED:raise ValueError('Only seed 20261101 is authorized')
        if self.task not in ('cv','nlp') or self.method not in METHODS:raise ValueError('Unknown task/method')
        if self.stage not in ('smoke','search'):raise ValueError('Unknown stage')
        if self.private!=(self.clip is not None) or self.scaled!=(self.eps_scale is not None):raise ValueError('Invalid clipping/scale configuration')
        if not all(math.isfinite(x) and x>0 for x in (self.lr,self.clip,self.eps_scale) if x is not None):raise ValueError('Nonpositive parameter')
    @property
    def private(self):return True
    @property
    def scaled(self):return True
    @property
    def mechanism(self):return self.method.rsplit('-',1)[1]
    @property
    def workload(self):return 'momentum-bias'
    @property
    def formal(self):return False
    @property
    def spacing(self):return 62 if self.task=='nlp' else 45
    @property
    def total_steps(self):return 5*self.spacing
    @property
    def physical_batch(self):return 250 if self.task=='cv' else 1000
    def values(self):return asdict(self)
    @property
    def id(self):
        return json_hash(self.values())[:20]
    @property
    def output(self):return RESULTS/self.stage/self.task/self.id
    def protocol(self):
        return dict(self.values(),epochs=5,total_steps=self.total_steps,logical_batch_size=1000,physical_batch_size=self.physical_batch,
            optimizer=dict(name='STP-EX',beta1=.9,beta2=.999,eps_scale=self.eps_scale,weight_decay=0,lr=self.lr),
            privacy=dict(epsilon=8.,delta=1e-5,k=5,b_participation=self.spacing,max_grad_norm=self.clip,
                adjacency='add_remove_zero_out',sampling_amplification=False),precision='FP32',full_finetuning=True,
            stp='exp15 original: previous vhat; scaled clip/noise; momentum of q; v from (denominator*q)^2; no final denominator',
            evaluation='internal',participation='fixed epoch repeated order',
            noise_seed=self.seed+1,dropout_seed='seed+100000+logical_step')
