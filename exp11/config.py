"""Fixed single-seed, internal-validation-only CIFAR-100 protocol."""
from pathlib import Path
import hashlib
import json
from exp11 import BASE, RESULTS

SEED = 20261101
SPLIT_SEED = 20261008
T = 225
CLIP = 24.
LR = .003
INITIAL_LAMBDAS = [0.] + [10. ** (k / 4) for k in range(-16, 13)]


def save_json(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(BASE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def lambda_tag(lam):
    return f'lambda_{lam:.17g}'


def trial_output(lam, smoke=False):
    return RESULTS / ('smoke' if smoke else 'cv') / lambda_tag(lam)


def protocol(lam, smoke=False):
    return dict(lambda_value=lam, seed=SEED, task='cv', stage='smoke' if smoke else 'search',
                epochs=5, total_steps=T, dataset_size=45000, validation_size=5000,
                physical_batch_size=250, logical_batch_size=1000, gradient_accumulation=4,
                optimizer=dict(name='torch.optim.Adam', beta1=.9, beta2=.999, eps=1e-8,
                               weight_decay=0., lr=LR), full_finetuning=True,
                privacy=dict(epsilon=8., delta=1e-5, k=5, b_participation=45,
                             max_grad_norm=CLIP, adjacency='add_remove_zero_out',
                             sampling_amplification=False),
                official_evaluation_used=False, image_size=224,
                rng='Exp9 initialization/order/augmentation; independent innovations seed+1',
                normalization='IID for both J1 and J2_mse', objective_C_clip=CLIP,
                privacy_scope='Per-training epsilon=8; internal validation assumed public. '
                              'No composed search privacy claim.')


def code_hashes():
    return {str(p.relative_to(BASE)): file_hash(p) for p in sorted(BASE.rglob('*.py'))
            if 'runtime' not in p.parts and 'results' not in p.parts}
