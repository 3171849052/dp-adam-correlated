"""Exp10's two independent methods and fixed 184-training protocol."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
from exp10 import ROOT, BASE, RESULTS
MODEL_PATH = ROOT / 'cache/bert-tiny'
SPLIT_SEED = 20261008
SEARCH_SEED = 20261101
RECHECK_SEEDS = (20261102, 20261103, 20261104)
FINAL_SEEDS = tuple(range(20261111, 20261121))
METHODS = ('adam-aware-mf', 'blockadam-mf')
LAMBDAS = (.1, .3, 1., 3.)
BLOCKS = (2, 5, 10, 25)
COUNTS = dict(search=120, recheck=24, final=40, total=184)
def save_json(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(BASE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
def array_hash(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
@dataclass(frozen=True)
class Trial:
    task: str
    method: str
    lr: float
    C: float
    module: float
    seed: int = SEARCH_SEED
    epsilon: float = 8.
    stage: str = 'search'
    smoke_phase: str = 'search'
    def __post_init__(self):
        assert self.task in ('cv','nlp') and self.method in METHODS
        assert self.stage in ('smoke','search','recheck','final')
        assert self.smoke_phase in ('search','final') and self.epsilon == 8.
        assert all(math.isfinite(v) and v > 0 for v in (self.lr,self.C))
        assert self.module in (LAMBDAS if self.method == METHODS[0] else BLOCKS)
        if self.stage == 'search': assert self.seed == SEARCH_SEED
        if self.stage == 'recheck': assert self.seed in RECHECK_SEEDS
        if self.stage == 'final': assert self.seed in FINAL_SEEDS
    @property
    def formal(self): return self.stage == 'final' or self.stage == 'smoke' and self.smoke_phase == 'final'
    @property
    def spacing(self): return 62 if self.task == 'nlp' else (50 if self.formal else 45)
    @property
    def total_steps(self): return 5*self.spacing
    @property
    def physical_batch(self): return 250 if self.task == 'cv' else 1000
    def values(self): return asdict(self)
    @property
    def id(self): return hashlib.sha256(json.dumps(self.values(),sort_keys=True).encode()).hexdigest()[:20]
    @property
    def output(self): return RESULTS / self.stage / self.task / self.id
    def protocol(self):
        return dict(self.values(), epochs=5, total_steps=self.total_steps, dataset_size=self.spacing*1000,
            physical_batch_size=self.physical_batch, logical_batch_size=1000, gradient_accumulation=1000//self.physical_batch,
            max_length=128 if self.task=='nlp' else None,image_size=224 if self.task=='cv' else None,
            optimizer=dict(beta1=.9,beta2=.999,eps=1e-8,weight_decay=0.,lr=self.lr),
            privacy=dict(epsilon=8.,delta=1e-5,k=5,b_participation=self.spacing,max_grad_norm=self.C,
                         adjacency='add_remove_zero_out',sampling_amplification=False),
            bandinvmf=dict(num_bands=4,noise='momentum_bias_bandinvmf',module=self.module),full_finetuning=True,
            privacy_assumption='Per-training epsilon=8 only. Internal validation assumed public; no pipeline composition claim.',
            rng='Exp9 initialization/order/augmentation/dropout; independent innovation seed+1')
