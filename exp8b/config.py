"""Locked SST-2 protocol and the seven independently searched methods."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
from exp8b import BASE, ROOT, RESULTS

MODEL_ID = 'prajjwal1/bert-tiny'
MODEL_REVISION = '6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837'
DATA_REVISION = 'bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c'
CHECKPOINT_SHA256 = 'dab2c2bddcfb48ea430ef63fd76d46d67d704487844d967256a50dd7d7fd0a66'
TRAIN_SHA256 = '66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229'
MODEL_PATH = ROOT/'cache/bert-tiny'
DATA_PATH = ROOT/'data/sst2/train'
SPLIT_SEED = 20261008
SEED = 20261001
FINAL_SEEDS = tuple(range(20261011, 20261021))
IID = 'dp-adam-iid'
SGD = 'dp-adam-sgd-bandinvmf'
MOMENTUM = 'dp-adam-momentum-bandinvmf'
BIAS = 'dp-adam-momentum-bias-bandinvmf'
SGD_SCALE, MOMENTUM_SCALE, BIAS_SCALE = (m+'-scale' for m in (SGD, MOMENTUM, BIAS))
METHODS = {IID: dict(geometry='standard', noise='iid')}
for m, n in ((SGD,'prefix_bandinvmf'),(MOMENTUM,'momentum_bandinvmf'),(BIAS,'momentum_bias_bandinvmf')):
    METHODS[m] = dict(geometry='standard',noise=n)
for m in (SGD,MOMENTUM,BIAS):
    METHODS[m+'-scale'] = dict(METHODS[m],geometry='scale')
LRS = (3e-5,1e-4,3e-4,1e-3)
STANDARD_CS = (.1,1.,10.,30.,100.)
SCALE_EPS = (.03,.1,.3,1.)

@dataclass(frozen=True)
class Config:
    method: str = IID
    seed: int = SEED
    lr: float = 1e-4
    C: float = 1.
    eps_scale: float | None = None
    physical_batch_size: int = 1000
    logical_batch_size: int = 1000
    max_length: int = 128
    epochs: int = 5
    total_steps: int = 310
    dataset_size: int = 62000
    beta1: float = .9
    beta2: float = .999
    adam_eps: float = 1e-8
    weight_decay: float = 0.
    dropout: float = .1

    def __post_init__(self):
        assert self.method in METHODS and self.seed in (SEED,*FINAL_SEEDS)
        assert (self.physical_batch_size,self.logical_batch_size,self.max_length)==(1000,1000,128)
        assert (self.epochs,self.total_steps,self.dataset_size)==(5,310,62000)
        assert (self.beta1,self.beta2,self.adam_eps,self.weight_decay,self.dropout)==(.9,.999,1e-8,0.,.1)
        assert all(math.isfinite(x) and x>0 for x in (self.lr,self.C))
        if self.geometry=='scale':
            assert self.eps_scale is not None and math.isfinite(self.eps_scale) and self.eps_scale>0
        else:
            assert self.eps_scale is None

    @property
    def geometry(self):
        return METHODS[self.method]['geometry']

    def protocol(self):
        return dict(asdict(self),model=MODEL_ID,task='GLUE SST-2',full_finetuning=True,
                    head='random_initialization',loss='cross_entropy',fixed_epoch_order=True,
                    privacy=dict(epsilon=8,delta=1e-5,adjacency='add_remove_zero_out',
                                 sampling_amplification=False,k=5,b_participation=62,max_grad_norm=self.C),
                    bandinvmf=dict(num_bands=4,noise=METHODS[self.method]['noise']),
                    rng='fixed split/order; paired initialization; dropout seed=seed+100000+step; independent noise seed=seed+1')

def trial(method,lr,C,eps_scale=None,seed=SEED):
    Config(method=method,seed=seed,lr=lr,C=C,eps_scale=eps_scale)
    return dict(method=method,seed=seed,lr=float(lr),C=float(C),eps_scale=eps_scale)

def trial_id(values):
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()[:16]

def trial_dir(values,category):
    assert category in ('smoke','search','final')
    return RESULTS/category/'trials'/trial_id(values)

def save_json(path,value):
    path=Path(path).resolve()
    assert path.is_relative_to(BASE)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
