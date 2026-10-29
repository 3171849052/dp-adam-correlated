"""Fixed experimental protocol and canonical training identity."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,math
from exp16c import ROOT,BASE,RESULTS
MODEL_PATH=ROOT/'cache/bert-tiny'
SPLIT_SEED=20261008
SEARCH_SEED=20261101
RECHECK_SEEDS=(20261102,20261103)
FINAL_SEEDS=tuple(range(20261111,20261121))
A_METHODS=('adam-mf-sgd-b2lt','adam-mf-momentum-bias-b2lt')
MOMENTUM_METHODS=('momentum-iid','momentum-mf-sgd-b2lt','momentum-mf-momentum-b2lt')
STPEX='stpex-mf-momentum-bias-b2lt'
HISTORICAL_METHODS=(*A_METHODS,'adam-mf-momentum-b2lt','adam-mf-sgd-stp-b2lt','adam-iid')
METHODS=(*HISTORICAL_METHODS,STPEX,*MOMENTUM_METHODS)
TUNED_METHODS=(*A_METHODS,STPEX,*MOMENTUM_METHODS)
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
    if not path.is_relative_to(BASE):raise ValueError(f'Output outside exp16c: {path}')
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
        if self.task!='cv' or self.method not in METHODS:raise ValueError('Unknown task/method')
        if self.stage not in ('smoke','search','review','final'):raise ValueError('Unknown stage')
        if self.private!=(self.clip is not None) or self.scaled!=(self.eps_scale is not None):raise ValueError('Invalid clipping/scale configuration')
        if not all(math.isfinite(x) and x>0 for x in (self.lr,self.clip,self.eps_scale) if x is not None):raise ValueError('Nonpositive parameter')
    @property
    def private(self):return self.method!='adam'
    @property
    def scaled(self):return '-stp-' in self.method or self.method.startswith('stpex')
    @property
    def mechanism(self):return self.method.rsplit('-',1)[1] if '-mf-' in self.method else 'iid'
    @property
    def workload(self):return self.method.split('-mf-',1)[1].removesuffix('-'+self.mechanism).removesuffix('-stp') if '-mf-' in self.method else 'identity'
    @property
    def formal(self):return self.stage=='final' or self.stage=='smoke' and self.smoke_phase=='final'
    @property
    def spacing(self):return 62 if self.task=='nlp' else 50 if self.formal else 45
    @property
    def total_steps(self):return 5*self.spacing
    @property
    def physical_batch(self):return 250 if self.task=='cv' else 1000
    def values(self):return asdict(self)
    @property
    def id(self):
        # Coarse and fine share stage=search and hence successful identical trials.
        return json_hash(self.values())[:20]
    @property
    def output(self):return RESULTS/self.stage/self.task/self.id
    def protocol(self):
        protocol = dict(self.values(),epochs=5,total_steps=self.total_steps,logical_batch_size=1000,physical_batch_size=self.physical_batch,
            optimizer=dict(name='Adam',beta1=.9,beta2=.999,eps=1e-8,weight_decay=0,lr=self.lr),
            privacy=dict(epsilon=8.,delta=1e-5,k=5,b_participation=self.spacing,max_grad_norm=self.clip,
                adjacency='add_remove_zero_out',sampling_amplification=False),precision='FP32',full_finetuning=True,
            stp='exp9 previous completed vhat; scaled clip/noise; undo scale; standard Adam eps=1e-8',
            evaluation='official' if self.stage=='final' else 'internal',participation='fixed epoch repeated order',
            noise_seed=self.seed+1,dropout_seed='seed+100000+logical_step')

        if self.method.startswith('momentum'):
            protocol['optimizer']=dict(name='Momentum',beta1=.9,bias_correction=False,nesterov=False,weight_decay=0,lr=self.lr)
        elif self.method.startswith('stpex'):
            protocol['optimizer']=dict(name='STP-EX',beta1=.9,beta2=.999,eps_scale=self.eps_scale,weight_decay=0,lr=self.lr)
            protocol['stp']='exp16b original: previous vhat; scaled clip/noise; momentum of q; v from (denominator*q)^2; no final denominator'
        return protocol
