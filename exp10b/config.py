"""Fixed lambda ablation: no learning-rate or clipping search."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
from exp10b import ROOT, BASE, RESULTS

MODEL_PATH = ROOT / 'cache/bert-tiny'
SPLIT_SEED = 20261008
LAMBDAS = (0., .01, .03, .1, .3, 1.)
STAGE1_SEEDS = tuple(range(20261101, 20261104))
STAGE2_SEEDS = tuple(range(20261104, 20261111))
FINAL_SEEDS = tuple(range(20261111, 20261121))
COUNTS = dict(stage1=36, stage2=84, stage4=40, total=160)

def save_json(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(BASE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Immutable evidence changed: {path}'
    else:
        save_json(path, value)

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
    module: float
    seed: int = 20261101
    stage: str = 'stage1'
    smoke_phase: str = 'search'

    def __post_init__(self):
        assert self.task in ('cv', 'nlp') and self.module in LAMBDAS
        assert self.stage in ('smoke', 'stage1', 'stage2', 'final')
        assert self.smoke_phase in ('search', 'final')
        if self.stage != 'smoke':
            assert self.smoke_phase == 'search'
            assert self.seed in dict(stage1=STAGE1_SEEDS, stage2=STAGE2_SEEDS, final=FINAL_SEEDS)[self.stage]

    @property
    def lr(self): return .003 if self.task == 'cv' else .005
    @property
    def C(self): return 30. if self.task == 'cv' else 1.
    @property
    def epsilon(self): return 8.
    @property
    def method(self): return 'adam-aware-mf'
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
    def id(self): return hashlib.sha256(json.dumps(self.values(), sort_keys=True).encode()).hexdigest()[:20]
    @property
    def output(self): return RESULTS / self.stage / self.task / self.id

    def protocol(self):
        return dict(self.values(), method=self.method, lr=self.lr, C=self.C, epsilon=8.,
            epochs=5, total_steps=self.total_steps, dataset_size=self.spacing*1000,
            physical_batch_size=self.physical_batch, logical_batch_size=1000,
            gradient_accumulation=1000//self.physical_batch,
            max_length=128 if self.task=='nlp' else None, image_size=224 if self.task=='cv' else None,
            optimizer=dict(beta1=.9, beta2=.999, eps=1e-8, weight_decay=0., lr=self.lr),
            privacy=dict(epsilon=8., delta=1e-5, k=5, b_participation=self.spacing,
                         max_grad_norm=self.C, adjacency='add_remove_zero_out', sampling_amplification=False),
            bandinvmf=dict(num_bands=4, noise='momentum_bias_bandinvmf', module=self.module),
            full_finetuning=True,
            privacy_assumption='Per-training epsilon=8 only; internal validation public/non-protected; no pipeline composition claim.',
            rng='Exp9 initialization/order/augmentation/dropout; common independent innovation seed+1')
