"""One fixed protocol; only C, lr, method and seed identify candidates."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

METHODS = ('dp-sgdm-iid', 'dp-sgdm-bandinvmf')
SEARCH_SEED = 20261001
FINAL_SEEDS = (20261011, 20261012, 20261013)
GPUS = (0, 1, 2)
FIXED = dict(dataset='CIFAR-100', data_root='data', download=False,
             model='vit_tiny_patch16_224', pretrained_root='cache',
             all_parameters_trainable=True, image_size=224,
             epochs=5, dataset_size=50000, logical_batch_size=1000,
             physical_batch_size=100, accumulation=10, steps_per_epoch=50,
             total_steps=250, beta=.9, weight_decay=0., epsilon=8., delta=1e-5,
             adjacency='replace_one', num_bands=4, workload='momentum',
             participation='fixed_epoch_sparse', sampling_amplification=False,
             clipping='per_example_global_l2_ghost',
             momentum_input='noisy_logical_average',
             fixed_epoch_order=True, noise_seed_offset=1, dtype='float32')


@dataclass(frozen=True)
class Trial:
    method: str
    seed: int
    lr: float
    C: float

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f'Unknown method: {self.method}')
        if not (math.isfinite(self.lr) and math.isfinite(self.C) and self.lr > 0 and self.C > 0):
            raise ValueError('lr and C must be finite and positive')

    @property
    def identity(self):
        payload = dict(fixed=FIXED, method=self.method, seed=self.seed,
                       lr=float(self.lr).hex(), C=float(self.C).hex())
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def asdict(self):
        return asdict(self)
