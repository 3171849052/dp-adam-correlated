"""Fixed protocol and seven canonical methods."""
import hashlib
import json
import math
from pathlib import Path
import yaml
from exp7b import BASE

SEED = 20261001
FINAL_SEEDS = tuple(range(20261011, 20261021))
IID = 'dp-adam-iid'
SGD = 'dp-adam-sgd-bandinvmf'
MOMENTUM = 'dp-adam-momentum-bandinvmf'
BIAS = 'dp-adam-momentum-bias-bandinvmf'
SGD_SCALE = SGD + '-scale'
MOMENTUM_SCALE = MOMENTUM + '-scale'
BIAS_SCALE = BIAS + '-scale'
METHODS = {IID: dict(geometry='standard', noise='iid')}
for method, noise in ((SGD, 'prefix_bandinvmf'), (MOMENTUM, 'momentum_bandinvmf'),
                      (BIAS, 'momentum_bias_bandinvmf')):
    METHODS[method] = dict(geometry='standard', noise=noise)
    METHODS[method + '-scale'] = dict(geometry='scale', noise=noise)
METHODS = {m: METHODS[m] for m in (IID, SGD, MOMENTUM, BIAS, SGD_SCALE, MOMENTUM_SCALE, BIAS_SCALE)}
SEARCH_METHODS = (SGD, MOMENTUM_SCALE, BIAS, BIAS_SCALE)
FROZEN_METHODS = (IID, MOMENTUM, SGD_SCALE)

def cell_for(method):
    return METHODS[method]

def validate(config):
    assert config['seed'] in (SEED, *FINAL_SEEDS) and config['data_root'] == 'data'
    assert (config['epochs'], config['dataset_size'], config['logical_batch_size'], config['physical_batch_size']) == (5, 50000, 1000, 250)
    o, p = config['optimizer'], config['privacy']
    assert (o['beta1'], o['beta2'], o['eps'], o['weight_decay']) == (.9, .999, 1e-8, 0)
    assert (p['epsilon'], p['delta'], p['adjacency'], p['sampling_amplification']) == (8, 1e-5, 'add_remove_zero_out', False)
    assert config['bandinvmf']['num_bands'] == 4
    assert config['model'] == dict(architecture='vit_tiny_patch16_224', pretrained=True, num_classes=100, image_size=224)
    assert math.isfinite(config['scale']['eps_scale']) and config['scale']['eps_scale'] > 0
    assert math.isfinite(o['lr']) and o['lr'] > 0
    assert math.isfinite(p['max_grad_norm']) and p['max_grad_norm'] > 0
    for key, value in (('k', 5), ('b_participation', 50)):
        assert p.get(key, value) == value
    assert config.get('total_steps', 250) == 250 and config.get('gradient_accumulation', 4) == 4
    config['gradient_accumulation'] = 4
    config['total_steps'] = 250
    p.update(k=5, b_participation=50)
    return config

def load_config(path=BASE / 'config.yaml'):
    return validate(yaml.safe_load(Path(path).read_text()))

def trial(method, lr, C, eps_scale=None, seed=SEED):
    assert method in METHODS and seed in (SEED, *FINAL_SEEDS)
    assert all(math.isfinite(float(x)) and x > 0 for x in (lr, C))
    if METHODS[method]['geometry'] == 'scale':
        assert eps_scale is not None and math.isfinite(eps_scale) and eps_scale > 0
    else:
        assert eps_scale is None
    return dict(method=method, seed=seed, lr=float(lr), C=float(C), eps_scale=None if eps_scale is None else float(eps_scale))

def trial_id(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]

def trial_dir(values, category):
    assert category in ('smoke', 'search', 'final')
    return BASE / 'results' / category / 'trials' / trial_id(values)

def save_json(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(BASE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
