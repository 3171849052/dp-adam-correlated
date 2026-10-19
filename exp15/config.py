"""Exp15 protocol; seeds are read from Exp9 source without importing Exp9."""
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from exp15 import ROOT, BASE, RESULTS

MODEL_PATH = ROOT / 'cache/bert-tiny'
METHODS = ('momentum', 'momentum-bias', 'stpex', 'stpex2', 'stpex-adam3')


def historical_seed(name):
    tree = ast.parse((ROOT/'exp9/config.py').read_text())
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    if isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id == 'tuple'
                and len(node.args) == 1 and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name) and node.args[0].func.id == 'range'):
            raise ValueError('Unsupported historical seed expression')
        return tuple(range(*(ast.literal_eval(a) for a in node.args[0].args)))
    return ast.literal_eval(node)


FINAL_SEEDS = historical_seed('FINAL_SEEDS')
SEARCH_SEED = historical_seed('SEARCH_SEED')
SPLIT_SEED = historical_seed('SPLIT_SEED')
PILOT_SEED = SEARCH_SEED - 1
SMOKE_SEED = SEARCH_SEED - 2


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def array_hash(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(BASE): raise ValueError('Exp15 writes must stay in exp15/')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def inherited_eps(task):
    # Fixed historical Momentum-Bias Scale denominator constant, not an Exp15 search.
    values = json.loads((ROOT/'exp9/results/frozen_configs.json').read_text())
    return float(values[task]['dp-adam-momentum-bias-bandinvmf-scale']['eps_scale'])


@dataclass(frozen=True)
class Trial:
    task: str
    method: str
    lr: float
    clip: float = 1.
    stage: str = 'smoke'
    seed: int = SMOKE_SEED
    smoke_phase: str = 'search'
    epsilon: float = 8.
    # An explicit replicate label changes output location but never RNG or identity.
    replicate: str = 'primary'

    def __post_init__(self):
        if self.task not in ('cv', 'nlp') or self.method not in METHODS: raise ValueError('Unknown task/method')
        if self.stage not in ('smoke', 'pilot', 'search', 'final'): raise ValueError('Unknown stage')
        if self.smoke_phase not in ('search', 'final'): raise ValueError('Unknown horizon')
        if self.replicate not in ('primary', 'repeat'): raise ValueError('Unknown replicate')
        if not all(math.isfinite(x) and x > 0 for x in (self.lr, self.clip)): raise ValueError('lr/clip must be positive and finite')
        if self.epsilon != 8: raise ValueError('Fixed epsilon=8 protocol')
        required = dict(smoke=(SMOKE_SEED,), pilot=(PILOT_SEED,), search=(SEARCH_SEED,), final=FINAL_SEEDS)
        if self.seed not in required[self.stage]: raise ValueError('Seed does not belong to stage')

    @property
    def scaled(self): return self.method.startswith('stpex')
    @property
    def formal(self): return self.stage == 'final' or self.stage == 'smoke' and self.smoke_phase == 'final'
    @property
    def spacing(self): return 62 if self.task == 'nlp' else (50 if self.formal else 45)
    @property
    def total_steps(self): return 5*self.spacing
    @property
    def physical_batch(self): return 250 if self.task == 'cv' else 1000
    @property
    def eps_scale(self): return inherited_eps(self.task) if self.scaled else None
    def values(self): return asdict(self)
    @property
    def id(self): return json_hash({k:v for k,v in self.values().items() if k != 'replicate'})[:20]
    @property
    def output(self): return RESULTS/self.stage/self.replicate/self.task/self.id

    def protocol(self):
        source = dict(momentum='none', **{'momentum-bias':'none', 'stpex':'g_tilde_squared',
                      'stpex2':'q_tilde_squared', 'stpex-adam3':'uncorrected_u_squared'})[self.method]
        return dict(**self.values(), epochs=5, total_steps=self.total_steps, dataset_size=1000*self.spacing,
                    logical_batch_size=1000, physical_batch_size=self.physical_batch,
                    image_size=224 if self.task=='cv' else None, max_length=128 if self.task=='nlp' else None,
                    optimizer=dict(name=self.method, beta1=.9, beta2=.999, weight_decay=0.,
                                   first_moment_bias_correction=self.method!='momentum',
                                   second_moment_bias_correction=self.scaled, second_moment_input=source,
                                   clipping_space='preconditioned' if self.scaled else 'original',
                                   eps_scale=self.eps_scale, initial_denominator=self.eps_scale if self.scaled else 1.,
                                   update='-lr * uhat' if self.scaled else '-lr * m(hat)'),
                    privacy=dict(epsilon=self.epsilon, delta=1e-5, k=5, b_participation=self.spacing,
                                 max_grad_norm=self.clip, adjacency='add_remove_zero_out', sampling_amplification=False),
                    noise=dict(family='four_band_Exp9_momentum_bias', controlled_filter=True,
                               momentum_filter_not_individually_optimized=self.method=='momentum'),
                    seeds=dict(final=list(FINAL_SEEDS), source='exp9/config.py',
                               source_sha256=file_hash(ROOT/'exp9/config.py')),
                    privacy_scope='single training only; internal validation assumed public; no search/repeat composition claim; raw diagnostics are private audit artifacts')
