"""The fixed 2x2 protocol and explicit tunable hyperparameters."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math

METHODS = ('dp-lora-iid', 'dp-lora-bandinvmf', 'dp-lora-iid-scale', 'dp-lora-bandinvmf-scale')
GPUS = (1, 2, 3)
SEARCH_SEED = 20261001
FINAL_SEEDS = (20261011, 20261012, 20261013)
FIXED = dict(protocol='exp6-lora-v1', dataset='CIFAR-100', dataset_size=50000,
    data_root='data', pretrained_root='cache', download=False,
    architecture='vit_tiny_patch16_224', image_size=224, classes=100,
    epochs=5, logical_batch_size=1000, steps_per_epoch=50, total_steps=250,
    rank=8, alpha=8, lora_modules=48, beta1=.9, beta2=.999, adam_eps=1e-8,
    weight_decay=0., epsilon=8., delta=1e-5, num_bands=4,
    workload='momentum_bandinvmf', adjacency='add_remove_zero_out',
    fixed_epoch_order=True, sampling_amplification=False,
    geometry='actual_weight', dtype='float32', noise_seed_offset=1,
    augmentation='sample_keyed_seed_epoch_index', backbone_frozen=True)
SMOKE_STEPS = 3


@dataclass(frozen=True)
class Trial:
    method: str
    seed: int = SEARCH_SEED
    lr: float = 1e-3
    C: float = 1.
    geom_eps: float = .1
    physical_batch_size: int = 8

    def __post_init__(self):
        assert self.method in METHODS
        assert all(math.isfinite(v) and v > 0 for v in (self.lr, self.C, self.geom_eps))
        assert self.physical_batch_size > 0 and 1000 % self.physical_batch_size == 0

    @property
    def scaled(self):
        return self.method.endswith('-scale')

    @property
    def noise(self):
        return 'momentum_bandinvmf' if 'bandinvmf' in self.method else 'iid'

    def asdict(self):
        return asdict(self)

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(dict(fixed=FIXED, trial=self.asdict()),
                                         sort_keys=True).encode()).hexdigest()
