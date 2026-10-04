"""Protocol constants; only learning rate and update clip norm are searched."""
from pathlib import Path
import yaml
from exp4.runtime import EXP

METHODS = ('dp-adam-iid-uc', 'dp-adam-bandinvmf-uc')
FIXED = dict(
    search_seed=20261001, final_seeds=[20261011, 20261012, 20261013],
    data_root='data', epochs=5, dataset_size=50000, logical_batch_size=1000,
    physical_batch_size=250, gradient_accumulation=4, total_steps=250,
    gpus=[0, 2, 3], max_parallel_trials=3,
    uc_includes_lr=False, uc_includes_sign=False, noise_before_lr=True,
    adam_state_uses_raw_batch_gradient=True, per_example_clipping=False,
    sampling_amplification=False, bandinvmf_workload='sgd', bandinvmf_num_bands=4,
    model=dict(architecture='vit_tiny_patch16_224', pretrained=True,
               num_classes=100, image_size=224),
    privacy=dict(epsilon=8.0, delta=1e-5, adjacency='replace_one',
                 participation='full_temporal'),
)


def load_config(path=EXP / 'config.yaml'):
    cfg = yaml.safe_load(Path(path).read_text())
    for key, expected in FIXED.items():
        assert cfg[key] == expected, f'Fixed protocol mismatch: {key}'
    opt = cfg['optimizer']
    assert {k: opt[k] for k in ('beta1', 'beta2', 'eps', 'weight_decay')} == dict(
        beta1=.9, beta2=.999, eps=1e-8, weight_decay=0.0)
    assert opt['lr'] > 0 and opt['update_clip_norm'] > 0
    return cfg
