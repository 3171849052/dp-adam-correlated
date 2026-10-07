"""Audit and import completed evidence; original experiments remain read-only."""
import json
import shutil
import numpy as np
import torch
import yaml
from exp7 import BASE, ROOT, runtime
from exp7.config import IID, MOMENTUM, SCALE, SEED, cell_for, trial, trial_id, validate, save_json
from exp2.bandinvmf import build_matrices, materialize
from exp2.model import checkpoint_path, checkpoint_sha256, pretrained_vit, initialization_digest
from exp2.privacy import calibrate
runtime()

def import_history():
    torch.set_num_threads(2)
    from exp7.train import seed_all
    seed_all(SEED)
    model, metadata = pretrained_vit()
    init = initialization_digest(model)
    order = torch.randperm(50000, generator=torch.Generator().manual_seed(SEED)).numpy()
    points = []
    for method, family in ((IID, 'iid'), (MOMENTUM, 'momentum')):
        for C in (1, 3, 10, 30, 100, 300):
            points.append((method, ROOT / f'exp2/results/matched_search/{family}_standard_matched/clip_{C}'))
    points.append((SCALE, ROOT / 'exp2/results/search/prefix_scale/anchor'))
    for lr in (.003, .007):
        points.append((MOMENTUM, ROOT / f'exp3b/results/tuning/adam_standard_lr_{lr}_clip_30'))
    rows = []
    for method, source in points:
        old = json.loads((source / 'summary.json').read_text())
        assert old['status'] == 'completed' and not old['smoke'] and old['seed'] == SEED
        assert old['optimizer_steps'] == old['noise_steps'] == 250
        is_exp2 = (source / 'config.yaml').exists()
        if is_exp2:
            cfg = yaml.safe_load((source / 'config.yaml').read_text())
            actual_init = old['initialization_sha256']
            checkpoint = old['pretrained_checkpoint_sha256']
            assert cfg['optimizer']['lr'] == old['lr']
            assert cfg['privacy']['max_grad_norm'] == old['max_grad_norm']
            assert (cfg['geometry'], cfg['noise']) == tuple(cell_for(method).values())
        else:
            resolved = json.loads((source / 'config.json').read_text())
            cfg = resolved['original_config']
            actual_init = resolved['initialization_sha256']
            checkpoint = resolved['checkpoint']['checkpoint_sha256']
            assert resolved['effective_steps'] == 250 and resolved['effective_logical_batch_size'] == 1000
            assert resolved['mf_workload_beta'] == .9 and resolved['spec']['method'] == 'momentum_standard'
            assert resolved['spec']['lr'] == old['lr'] and resolved['spec']['max_grad_norm'] == old['max_grad_norm']
            assert resolved['spec']['seed'] == SEED
        validate(cfg)
        assert actual_init == init and checkpoint == metadata['checkpoint_sha256']
        assert old['epochs'][-1]['epoch'] == 5 and len(old['epochs']) == 5
        np.testing.assert_array_equal(np.load(source / 'train_order.npy'), order)
        eps = old['eps_scale'] if method == SCALE else None
        values = trial(method, old['lr'], old['max_grad_norm'], eps)
        cfg['privacy']['max_grad_norm'] = values['C']
        coefficients, strategy, workload = build_matrices(cell_for(method)['noise'], 250, 4, .9)
        privacy = calibrate(strategy, cfg)
        matrices = np.load(source / 'matrices.npz')
        np.testing.assert_allclose(matrices['noising_coefficients' if is_exp2 else 'coefficients'], coefficients, rtol=1e-12)
        np.testing.assert_allclose(matrices['strategy'], strategy, rtol=1e-12)
        np.testing.assert_allclose(matrices['W'], materialize(workload, 250), rtol=1e-12)
        assert np.isclose(matrices['innovation_std_sum'], privacy['innovation_std_sum'], rtol=1e-12)
        for name in ('metrics.csv', 'train.log', 'train_order.npy', 'matrices.npz', 'final.pt'):
            assert (source / name).is_file()
        directory = BASE / 'results/trials' / trial_id(values)
        if directory.exists():
            row = json.loads((directory / 'summary.json').read_text())
            assert all(row[k] == v for k, v in values.items()) and row['status'] == 'completed'
        else:
            directory.mkdir(parents=True)
            for f in source.iterdir():
                if f.is_file() and f.name != 'summary.json':
                    shutil.copy2(f, directory / f.name)
            save_json(directory / 'historical_summary.json', old)
            if not is_exp2:
                yaml.safe_dump(dict(cfg, method=method, historical_config=str(source / 'config.json')), (directory / 'config.yaml').open('w'))
            row = dict(old, **cell_for(method), schema_version=1, C=values['C'],
                       final_test_top1=old['epochs'][-1]['test_top1'], source='historical',
                       result_dir=str(directory.relative_to(ROOT)), historical_result_dir=str(source.relative_to(ROOT)))
            row.update(values)
            save_json(directory / 'summary.json', row)
        if is_exp2:
            shutil.copy2(source / 'config.yaml', directory / 'historical_config.yaml')
        cfg['optimizer']['lr'] = values['lr']
        canonical = dict(cfg, method=method, **cell_for(method), source='historical',
                         historical_result_dir=str(source.relative_to(ROOT)))
        with (directory / 'config.yaml').open('w') as f:
            yaml.safe_dump(canonical, f, sort_keys=False)
        rows.append(row)
    save_json(BASE / 'results/history_audit.json', dict(status='passed', imported=len(rows),
              initialization_sha256=init, pretrained_checkpoint_sha256=metadata['checkpoint_sha256'],
              checks=['fixed protocol', 'completed epoch 5', 'seed', 'local checkpoint', 'initialization',
                      'train order', 'noising coefficients', 'strategy', 'workload', 'calibration', 'artifacts']))
    return rows
