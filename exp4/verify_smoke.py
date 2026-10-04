"""Audit the two completed local-data smoke trials; write evidence in exp4."""
from exp4.runtime import ROOT, EXP
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from exp4.config import METHODS, load_config


def main():
    load_config()
    evidence = {}
    inits, heads, orders, augmentations, first_steps = set(), set(), [], set(), []
    for method in METHODS:
        folder = EXP / 'results/smoke' / method / 'seed_20261001'
        cfg = yaml.safe_load((folder / 'config.yaml').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        steps = list(csv.DictReader((folder / 'steps.csv').open()))
        assert summary['status'] == 'completed' and summary['smoke']
        assert summary['optimizer_steps'] == summary['noise_steps'] == len(steps) == 2
        assert summary['physical_batches'] == 8 and cfg['physical_batch_size'] == 250
        assert cfg['visible_devices'] in ('0', '2', '3')
        assert cfg['local_data_root'] == str(ROOT / 'data') and not cfg['download']
        assert (ROOT / cfg['pretrained_cfg']['checkpoint_path']).resolve().is_relative_to(ROOT / 'cache')
        assert cfg['privacy_calibration']['total_steps'] == 250
        assert cfg['privacy_calibration']['per_step_sensitivity'] == 2 * cfg['optimizer']['update_clip_norm']
        matrices = np.load(folder / 'matrices.npz')
        np.testing.assert_array_equal(matrices['workload_coefficients'], np.ones(250))
        np.testing.assert_array_equal(matrices['W'], np.tril(np.ones((250, 250))))
        np.testing.assert_allclose(matrices['D'] @ matrices['strategy'], np.eye(250), atol=1e-12)
        for row in steps:
            raw, clipped, scale = (float(row[k]) for k in ('raw_update_norm', 'clipped_update_norm', 'clip_scale'))
            assert clipped <= cfg['optimizer']['update_clip_norm'] * (1 + 1e-5)
            assert np.isclose(clipped, raw * scale, rtol=1e-5)
            assert np.isclose(float(row['parameter_noise_std']), cfg['optimizer']['lr'] * float(row['noise_marginal_std']))
            assert np.isclose(float(row['signal_parameter_norm']), cfg['optimizer']['lr'] * clipped)
            assert all(np.isfinite(float(value)) for key, value in row.items() if value)
        inits.add(cfg['initialization_sha256'])
        heads.add(cfg['classifier_initialization_sha256'])
        orders.append(np.load(folder / 'train_order.npy'))
        augmentations.add(summary['final']['augmentation_trace_sha256'])
        first_steps.append(steps[0])
        evidence[method] = dict(steps=2, physical_batches=8, visible_gpu=cfg['visible_devices'],
                                checkpoint=cfg['pretrained_cfg'], calibration=summary['calibration'],
                                final=summary['final'], clipping_and_local_inputs_verified=True)
    assert len(inits) == len(heads) == len(augmentations) == 1
    np.testing.assert_array_equal(*orders)
    for key in ('train_loss', 'raw_update_norm', 'clipped_update_norm', 'clip_scale',
                'adam_m_norm', 'vhat_min', 'vhat_max', 'vhat_mean', 'vhat_rms'):
        assert first_steps[0][key] == first_steps[1][key], key
    evidence['paired_initialization_data_order_augmentation'] = True
    evidence['first_step_raw_adam_and_uc_identical_before_noise'] = True
    (EXP / 'results/smoke_verification.json').write_text(json.dumps(evidence, indent=2, allow_nan=False))
    print('Smoke audit passed: two methods × 2 logical steps; local data/cache; prefix workload; UC diagnostics.')


if __name__ == '__main__':
    main()
