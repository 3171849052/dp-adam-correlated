"""Fixed coordinate-normalized SGDM protocol; only tau and lr are searched."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

METHODS = ('dp-coordnorm-sgdm-iid', 'dp-coordnorm-sgdm-bandinvmf')
SEARCH_SEED = 20261001
FINAL_SEEDS = (20261011, 20261012, 20261013)
GPUS = (0, 1, 2)
CHUNK_SIZE = 50
FIXED = dict(version='coordnorm-v1', dataset='CIFAR-100', data_root='data', download=False,
             model='vit_tiny_patch16_224', pretrained_root='cache',
             all_parameters_trainable=True, image_size=224,
             epochs=5, dataset_size=50000, logical_batch_size=1000,
             physical_batch_size=100, accumulation=10, steps_per_epoch=50,
             total_steps=250, beta=.9, weight_decay=0., epsilon=8., delta=1e-5,
             adjacency='replace_one', num_bands=4, workload='momentum',
             participation='fixed_epoch_sparse', sampling_amplification=False,
             transform='per_example_coordinate_g/(abs(g)+tau)',
             clipping='global_l2_after_coordinate_transform', C=1.,
             momentum_input='noisy_logical_average', adam_hidden_state=False,
             fixed_epoch_order=True, noise_seed_offset=1, dtype='float32',
             augmentation='deterministic_per_seed_epoch_sample')

@dataclass(frozen=True)
class Trial:
    method: str
    seed: int
    lr: float
    tau: float
    C: float = 1.

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f'Unknown method: {self.method}')
        if not all(math.isfinite(v) and v > 0 for v in (self.lr, self.tau)):
            raise ValueError('lr and tau must be finite and positive')
        if self.C != 1.:
            raise ValueError('Exp5 fixes transformed global clip C=1')

    @property
    def identity(self):
        payload = dict(fixed=FIXED, method=self.method, seed=self.seed,
                       lr=float(self.lr).hex(), tau=float(self.tau).hex(), C=float(self.C).hex())
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def asdict(self):
        return asdict(self)
