"""Exp15c fixed protocol and content-addressed trial identities."""
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from exp15c import ROOT, BASE, RESULTS

METHODS = ('adam-no-noise', 'momentum-no-noise', 'momentum-mf-native')
MODEL_PATH = ROOT/'cache/bert-tiny'
SEARCH_SEED = 20261101
REVIEW_SEEDS = (20261102, 20261103)
FINAL_SEEDS = tuple(range(20261111, 20261121))
SMOKE_SEED = SEARCH_SEED-2
_tree = ast.parse((ROOT/'exp9/config.py').read_text())
SPLIT_SEED = next(ast.literal_eval(n.value) for n in _tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id=='SPLIT_SEED' for t in n.targets))


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def array_hash(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def save_json(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(BASE): raise ValueError('Writes must stay in exp15c/')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


@dataclass(frozen=True)
class Trial:
    task: str
    method: str
    lr: float
    clip: float | None = None
    stage: str = 'smoke'
    seed: int = SMOKE_SEED
    smoke_phase: str = 'search'
    replicate: str = 'primary'

    def __post_init__(self):
        if self.task not in ('cv','nlp') or self.method not in METHODS: raise ValueError('Unknown task/method')
        if self.lr<=0: raise ValueError('Positive learning rate required')
        if self.private != (self.clip is not None): raise ValueError('Only DP uses a clipping threshold')
        if self.private and self.clip<=0: raise ValueError('Positive clipping threshold required')
        if self.seed not in dict(smoke=(SMOKE_SEED,),search=(SEARCH_SEED,),review=REVIEW_SEEDS,final=FINAL_SEEDS)[self.stage]:
            raise ValueError('Seed/stage mismatch')
        if self.smoke_phase not in ('search','final') or self.replicate not in ('primary','repeat'): raise ValueError('Invalid smoke/replicate')

    @property
    def private(self): return self.method=='momentum-mf-native'
    @property
    def formal(self): return self.stage=='final' or self.stage=='smoke' and self.smoke_phase=='final'
    @property
    def spacing(self): return 62 if self.task=='nlp' else 50 if self.formal else 45
    @property
    def total_steps(self): return 5*self.spacing
    @property
    def physical_batch(self): return 250 if self.task=='cv' else 1000
    def values(self): return asdict(self)
    @property
    def id(self): return json_hash({k:v for k,v in self.values().items() if k!='replicate'})[:20]
    @property
    def output(self): return RESULTS/self.stage/self.replicate/self.task/self.id

    def protocol(self):
        return dict(**self.values(),epochs=5,total_steps=self.total_steps,logical_batch_size=1000,
            physical_batch_size=self.physical_batch,full_parameter_fp32=True,
            optimizer=dict(name=self.method,beta1=.9,beta2=.999,weight_decay=0,eps=1e-8 if self.method=='adam-no-noise' else None,
                           first_moment_bias_correction=self.method=='adam-no-noise'),
            privacy=dict(private=self.private,epsilon=8. if self.private else None,delta=1e-5 if self.private else None,
                         k=5,b_participation=self.spacing,max_grad_norm=self.clip,adjacency='add_remove_zero_out',sampling_amplification=False),
            clipping_space='original' if self.private else None,
            privacy_scope='single training only; internal validation assumed public; no search/repeat composition guarantee; raw diagnostics private audit artifacts')
