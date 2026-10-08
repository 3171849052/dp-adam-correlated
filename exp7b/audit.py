"""Content hashes and protocol/finite/FIFO/pairing assertions."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
import torch
from exp7b import BASE, ROOT
from exp7b.config import validate, save_json, cell_for
from exp7b.bandinvmf import build_matrices, materialize
from exp2.privacy import calibrate, epsilon_from_mu


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(value.tobytes()).hexdigest()


def audit_trial(directory):
    directory = Path(directory)
    assert directory.resolve().is_relative_to(BASE)
    row = json.loads((directory / 'summary.json').read_text())
    assert row['status'] == 'completed'
    cfg = validate(yaml.safe_load((directory / 'config.yaml').read_text()))
    assert cfg['seed'] == row['seed']
    assert (cfg['optimizer']['lr'], cfg['privacy']['max_grad_norm']) == (row['lr'], row['C'])
    assert all(cfg[k] == v for k, v in cell_for(row['method']).items())
    if row['eps_scale'] is not None:
        assert cfg['scale']['eps_scale'] == row['eps_scale']
    expected = 1 if row['smoke'] else 250
    epochs = 1 if row['smoke'] else 5
    assert row['optimizer_steps'] == row['noise_steps'] == expected
    assert row['planned_total_steps'] == 250 and row['physical_batches'] == 4 * expected
    metrics = list(csv.DictReader((directory / 'metrics.csv').open()))
    assert len(metrics) == epochs and int(metrics[-1]['logical_steps']) == expected
    assert [int(r['epoch']) for r in metrics] == list(range(1, epochs + 1))
    assert [int(r['logical_steps']) for r in metrics] == ([1] if row['smoke'] else [50,100,150,200,250])
    assert len(row['epochs']) == epochs and row['final_test_top1'] == float(metrics[-1]['test_top1'])
    assert all(int(r['train_examples']) == (1000 if row['smoke'] else 50000) for r in metrics)
    assert all(np.isfinite(float(r[k])) for r in metrics for k in ('train_loss', 'test_loss', 'test_top1', 'gdp_epsilon', 'noise_std'))
    d, strategy, W = build_matrices(cfg['noise'], 250, 4, .9)
    privacy = calibrate(strategy, cfg)
    matrices = np.load(directory / 'matrices.npz')
    for key, value in [('strategy', strategy), ('W', W), ('noising_coefficients', d),
                       ('M', materialize(d, 250) * privacy['innovation_std_sum'])]:
        np.testing.assert_allclose(matrices[key], value, rtol=1e-11, atol=1e-12)
        assert np.isfinite(matrices[key]).all()
    assert np.isclose(row['calibration']['innovation_std_sum'], privacy['innovation_std_sum'])
    assert row['calibration']['adjacency'] == 'add_remove_zero_out'
    assert (row['calibration']['k'], row['calibration']['b_participation'], row['calibration']['sampling_amplification']) == (5,50,False)
    assert np.isclose(epsilon_from_mu(cfg['privacy']['max_grad_norm'] * privacy['sensitivity'] / privacy['innovation_std_sum'], 1e-5), 8)
    if not row['smoke']:
        assert np.isclose(float(metrics[-1]['gdp_epsilon']), 8)
    for key, name in [('matrix_sha256', 'matrices.npz'), ('checkpoint_sha256', 'final.pt'), ('train_order_sha256', 'train_order.npy')]:
        assert row[key] == file_hash(directory / name)
    assert row['workload_sha256'] == array_hash(W) and row['strategy_sha256'] == array_hash(strategy)
    np.testing.assert_array_equal(np.load(directory / 'train_order.npy'),
                                  torch.randperm(50000,generator=torch.Generator().manual_seed(row['seed'])).numpy())
    assert row['augmentation_trace_sha256'] == [r['augmentation_trace_sha256'] for r in metrics]
    from exp2.model import checkpoint_path, checkpoint_sha256
    from exp7b import runtime
    runtime()
    assert row['pretrained_checkpoint_sha256'] == checkpoint_sha256(checkpoint_path())
    if not row['smoke'] and row['seed'] != 20261001:
        assert row['frozen_configs_sha256'] == file_hash(BASE / 'results/frozen_configs.json')
    checkpoint = torch.load(directory / 'final.pt', map_location='cpu', weights_only=True)
    assert checkpoint.get('logical_steps', expected) == expected
    assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
    for state in checkpoint['optimizer']['state'].values():
        assert int(state['step']) == expected
        assert all(torch.isfinite(state[k]).all() for k in ('exp_avg', 'exp_avg_sq'))
    if row.get('source') == 'new':
        assert row['finite'] is True and row['completed_epochs'] == epochs
    return row


def audit_pairing(rows):
    for seed in {r['seed'] for r in rows}:
        group = [r for r in rows if r['seed'] == seed]
        for key in ('initialization_sha256', 'classifier_initialization_sha256',
                    'pretrained_checkpoint_sha256', 'train_order_sha256', 'augmentation_trace_sha256'):
            assert all(r[key] == group[0][key] for r in group), (seed, key)


def audit_fifo():
    live, starts = {}, []
    path = BASE / 'results/scheduler.jsonl'
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        assert row['gpu'] in (1, 2, 3)
        if row['event'] == 'start':
            assert row['gpu'] not in live
            live[row['gpu']] = row['trial_id']
            starts.append((row['category'], row['trial_id']))
        else:
            assert live.pop(row['gpu']) == row['trial_id']
        assert len(live) <= 3
    assert not live and len(starts) == len(set(starts))
    return dict(status='passed', launches=len(starts), gpus=[1, 2, 3])

if __name__ == '__main__':
    rows = [audit_trial(p.parent) for p in (BASE / 'results').glob('*/trials/*/summary.json')
            if json.loads(p.read_text())['status'] == 'completed']
    for smoke in (True, False):
        audit_pairing([r for r in rows if r['smoke'] == smoke])
    save_json(BASE / 'results/artifact_audit.json', dict(audit_fifo(), audited_trials=len(rows)))
