"""Fixed protocol and canonical identities; no final-seed entry point."""
import hashlib
import json
import math
from pathlib import Path
import yaml
from exp7 import BASE

SEED = 20261001
METHODS = {
    'dp-adam-iid': dict(geometry='standard', noise='iid'),
    'dp-adam-momentum-bandinvmf': dict(geometry='standard', noise='momentum_bandinvmf'),
    'dp-adam-sgd-bandinvmf-scale': dict(geometry='scale', noise='prefix_bandinvmf'),
}
IID, MOMENTUM, SCALE = METHODS

def cell_for(method):
    return METHODS[method]

def validate(config):
    assert config['seed'] == SEED and config['data_root'] == 'data'
    assert (config['epochs'], config['dataset_size'], config['logical_batch_size'], config['physical_batch_size']) == (5, 50000, 1000, 250)
    opt, privacy = config['optimizer'], config['privacy']
    assert (opt['beta1'], opt['beta2'], opt['eps'], opt['weight_decay']) == (.9, .999, 1e-8, 0)
    assert (privacy['epsilon'], privacy['delta'], privacy['adjacency'], privacy['sampling_amplification']) == (8, 1e-5, 'add_remove_zero_out', False)
    assert config['bandinvmf']['num_bands'] == 4
    assert config['model'] == dict(architecture='vit_tiny_patch16_224', pretrained=True, num_classes=100, image_size=224)
    assert math.isfinite(config['scale']['eps_scale']) and config['scale']['eps_scale'] > 0
    config['gradient_accumulation'] = 4
    privacy.update(k=5, b_participation=50)
    config['total_steps'] = 250
    return config

def load_config(path=BASE / 'config.yaml'):
    return validate(yaml.safe_load(Path(path).read_text()))

def trial(method, lr, C, eps_scale=None):
    assert method in METHODS
    assert all(math.isfinite(float(x)) and x > 0 for x in (lr, C))
    if method == SCALE:
        assert eps_scale is not None and math.isfinite(eps_scale) and eps_scale > 0
    else:
        assert eps_scale is None
    return dict(method=method, seed=SEED, lr=float(lr), C=float(C), eps_scale=eps_scale)

def trial_id(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
