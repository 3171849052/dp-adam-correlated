"""Reuse only compatible completed Exp2/Exp7 trials, with local copies and audits."""
from functools import lru_cache
import json
import shutil
import numpy as np
import torch
import yaml
from exp7b import BASE, ROOT, runtime
from exp7b.config import METHODS, SEED, validate, trial, trial_id, trial_dir, save_json
from exp7b.bandinvmf import build_matrices, materialize
from exp7b.audit import file_hash, array_hash, audit_trial
from exp2.model import pretrained_vit, initialization_digest
from exp2.privacy import calibrate
runtime()

@lru_cache(None)
def reference():
    # Preserve caller RNG state; historical verification must not affect training.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model, metadata = pretrained_vit()
        return initialization_digest(model), initialization_digest(model.head), metadata['checkpoint_sha256']

@lru_cache(None)
def index():
    found = {}
    for root in (ROOT / 'exp7/results/trials', ROOT / 'exp2/results/search', ROOT / 'exp2/results/matched_search'):
        for path in sorted(root.glob('**/summary.json')):
            old = json.loads(path.read_text())
            if old.get('status') != 'completed' or old.get('smoke') or old.get('seed') != SEED:
                continue
            cfg_path = path.parent / 'config.yaml'
            if not cfg_path.exists():
                continue
            cfg = yaml.safe_load(cfg_path.read_text())
            method = next((m for m, cell in METHODS.items() if all(cfg.get(k) == v for k, v in cell.items())), None)
            if method is None:
                continue
            values = trial(method, old['lr'], old.get('C', old['max_grad_norm']),
                           old.get('eps_scale') if METHODS[method]['geometry'] == 'scale' else None)
            found.setdefault(trial_id(values), (path.parent, old, cfg))
    return found

def reuse(values):
    directory = trial_dir(values, 'search')
    if directory.exists() or trial_id(values) not in index():
        return False
    source, old, cfg = index()[trial_id(values)]
    validate(cfg)
    assert old['optimizer_steps'] == old['noise_steps'] == 250 and len(old['epochs']) == 5
    assert (cfg['optimizer']['lr'], cfg['privacy']['max_grad_norm']) == (values['lr'], values['C'])
    if values['eps_scale'] is not None:
        assert cfg['scale']['eps_scale'] == values['eps_scale']
    init, head, checkpoint = reference()
    assert old['initialization_sha256'] == init and old['pretrained_checkpoint_sha256'] == checkpoint
    assert old['classifier_initialization_sha256'] == head
    order = torch.randperm(50000, generator=torch.Generator().manual_seed(SEED)).numpy()
    np.testing.assert_array_equal(np.load(source / 'train_order.npy'), order)
    d, strategy, W = build_matrices(cfg['noise'], 250, 4, .9)
    privacy = calibrate(strategy, cfg)
    matrices = np.load(source / 'matrices.npz')
    for key, value in (('noising_coefficients', d), ('strategy', strategy), ('W', W),
                       ('M', materialize(d, 250) * privacy['innovation_std_sum'])):
        np.testing.assert_allclose(matrices[key], value, rtol=1e-11, atol=1e-12)
    assert old['epochs'][-1]['logical_steps'] == 250
    directory.mkdir(parents=True)
    for name in ('metrics.csv', 'mechanism_metrics.csv', 'train.log', 'train_order.npy', 'matrices.npz', 'final.pt'):
        shutil.copy2(source / name, directory / name)
    shutil.copy2(source / 'config.yaml', directory / 'historical_config.yaml')
    cfg.update(method=values['method'], source='historical')
    (directory / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    row = dict(old, **values, source='historical', finite=True, completed_epochs=5,
               final_test_top1=old['epochs'][-1]['test_top1'],
               historical_result_dir=str(source.relative_to(ROOT)),
               result_dir=str(directory.relative_to(ROOT)),
               workload_sha256=array_hash(W), strategy_sha256=array_hash(strategy))
    for key, name in (('matrix_sha256', 'matrices.npz'), ('checkpoint_sha256', 'final.pt'), ('train_order_sha256', 'train_order.npy')):
        row[key] = file_hash(directory / name)
    save_json(directory / 'historical_summary.json', old)
    save_json(directory / 'summary.json', row)
    audit_trial(directory)
    return True
