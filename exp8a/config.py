"""Immutable experiment protocol; only compute and shared pilot parameters vary."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from exp8a import BASE, ROOT, RESULTS

MODEL_ID = 'prajjwal1/bert-tiny'
MODEL_REVISION = '6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837'
DATA_REVISION = 'bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c'
CHECKPOINT_SHA256 = 'dab2c2bddcfb48ea430ef63fd76d46d67d704487844d967256a50dd7d7fd0a66'
TRAIN_SHA256 = '66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229'
MODEL_PATH = ROOT/'cache/bert-tiny'
DATA_PATH = ROOT/'data/sst2/train'
SPLIT_SEED = 20261008
SEEDS = (20261011, 20261012, 20261013)
LENGTHS = (32, 64, 128)
BATCHES = (4, 8, 20, 40, 50, 100, 125, 200, 250, 500, 1000)

@dataclass(frozen=True)
class Config:
    physical_batch_size: int = 20
    max_length: int = 64
    seed: int = SEEDS[0]
    lr: float = 5e-4
    C: float = 1.0
    eps_scale: float = .1
    logical_batch_size: int = 1000
    epochs: int = 5
    total_steps: int = 310
    dataset_size: int = 62000
    beta1: float = .9
    beta2: float = .999
    adam_eps: float = 1e-8
    dropout: float = 0.0

    def __post_init__(self):
        assert self.physical_batch_size in BATCHES and 1000 % self.physical_batch_size == 0
        assert self.max_length in LENGTHS
        assert (self.logical_batch_size, self.epochs, self.total_steps, self.dataset_size) == (1000,5,310,62000)
        assert (self.beta1, self.beta2, self.adam_eps, self.dropout) == (.9,.999,1e-8,0.)
        assert all(0 < v < float('inf') for v in (self.lr,self.C,self.eps_scale))

    def protocol(self):
        return dict(asdict(self), model=MODEL_ID, task='GLUE SST-2', loss='cross_entropy',
                    full_finetuning=True, head='random_initialization', padding='max_length',
                    fixed_epoch_order=True, official_validation_used=False,
                    privacy=dict(epsilon=8,delta=1e-5,adjacency='add_remove_zero_out',
                                 sampling_amplification=False,k=5,b_participation=62,max_grad_norm=self.C),
                    bandinvmf=dict(num_bands=4,workload='full_non_toeplitz_momentum_bias',beta1=.9),
                    rng='fixed split/order; paired initialization seed; independent noise seed+1; dropout=0 for partition invariance')

def save_json(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(BASE)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def file_hash(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
