"""Audit real GPU smoke results and their privacy/noise/geometry artifacts."""
from exp6.runtime import EXP, require_curve
import json
import math
import numpy as np
from exp6.config import METHODS, Trial, FIXED, SMOKE_STEPS, GPUS
from exp6.mechanism import matrices, materialize
from exp6.privacy import calibration, spent
from exp6.launch_batch import smoke_jobs, read_completed


def assert_finite(value):
    if isinstance(value, dict):
        for v in value.values(): assert_finite(v)
    elif isinstance(value, list):
        for v in value: assert_finite(v)
    elif isinstance(value, float):
        assert math.isfinite(value)


def verify():
    summaries, configs, arrays = [], [], []
    for job in smoke_jobs():
        folder = EXP / 'results/smoke' / job['trial']['method']
        summary = read_completed(job)
        config = json.loads((folder / 'config.json').read_text())
        assert config['fixed'] == FIXED and config['physical_gpu'] in GPUS
        trial = Trial(**config['trial'])
        expected_coef, strategy, workload = matrices(trial)
        privacy = calibration(strategy, trial)
        assert config['privacy'] == privacy
        assert abs(spent(strategy,250,trial,privacy)['epsilon']-8) < 1e-8
        array = dict(np.load(folder / 'matrices.npz'))
        for name, expected in [('coefficients',expected_coef),('strategy',strategy),('workload_coefficients',workload),
                               ('noising_matrix',materialize(expected_coef,250)),('workload',materialize(workload,250))]:
            np.testing.assert_array_equal(array[name],expected)
        assert float(array['innovation_std_sum']) == privacy['innovation_std_sum']
        assert all(np.isfinite(v).all() for v in array.values())
        rows = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
        geometry = [json.loads(line) for line in (folder / 'geometry.jsonl').read_text().splitlines()]
        assert len(rows) == len(geometry) == SMOKE_STEPS
        for i,(r,g) in enumerate(zip(rows,geometry),1):
            assert r['step'] == g['step'] == r['noise_draws'] == i
            assert r['train_examples'] == 1000
            assert 0 <= r['clipping_fraction'] <= 1 and 0 <= r['train_top1'] <= 1
            assert 0 < r['clip_factor_min'] <= r['clip_factor_max'] <= 1
            assert r['query_norm'] <= trial.C * (1+1e-5)
            assert r['noise_std_space'] == ('scaled_average' if trial.scaled else 'raw_average')
            expected_std = privacy['innovation_std_sum'] * np.linalg.norm(expected_coef[:i]) / 1000
            assert abs(r['noise_std']-expected_std) < 1e-12
            assert r['mu'] == spent(strategy,i,trial,privacy)['mu']
            assert ('scaled_per_example_norm_mean' in r) == trial.scaled
            assert len(g['layers']) == 48
            for layer in g['layers']:
                assert layer['PA_eig_min'] >= trial.geom_eps * (1-1e-5)
                assert layer['PB_eig_min'] >= trial.geom_eps * (1-1e-5)
                assert layer['PA_condition'] >= 1 and layer['PB_condition'] >= 1
            assert_finite(r); assert_finite(g)
        assert_finite(summary); assert_finite(config)
        import torch
        checkpoint = torch.load(folder / 'final.pt', map_location='cpu', weights_only=False)
        assert list(checkpoint['trainable']) == config['trainable_names']
        for tensor in checkpoint['trainable'].values(): assert torch.isfinite(tensor).all()
        for state in checkpoint['optimizer']['state'].values():
            for tensor in state.values(): assert torch.isfinite(tensor).all()
        summaries.append(summary); configs.append(config); arrays.append(array)
    for key in ('initialization_sha256','order_sha256'):
        assert len({s[key] for s in summaries}) == 1
    for key in ('augmentation_sha256','innovation_sha256'):
        assert all(s[key] == summaries[0][key] for s in summaries)
    for key in ('coefficients','strategy','workload_coefficients','workload','noising_matrix'):
        np.testing.assert_array_equal(arrays[1][key],arrays[3][key])
    np.testing.assert_array_equal(arrays[0]['strategy'], np.eye(250))
    assert all(c['gaussian_seed'] == configs[0]['gaussian_seed'] for c in configs)
    order = np.load(EXP / 'results/smoke' / METHODS[0] / 'train_order.npy')
    np.testing.assert_array_equal(np.sort(order),np.arange(50000))
    for m in METHODS[1:]:
        np.testing.assert_array_equal(order,np.load(EXP / 'results/smoke' / m / 'train_order.npy'))
    result = dict(status='passed', methods=list(METHODS), physical_gpus=list(GPUS),
        logical_steps_per_method=SMOKE_STEPS, examples_per_step=1000,
        same_initialization=True, same_order=True, same_augmentations=True,
        identical_standardized_innovations=True, identical_bandinvmf_geometry=True,
        full_trajectory_target=dict(epsilon=8., delta=1e-5),
        smoke_spent={s['trial']['method']:s['final_privacy'] for s in summaries})
    (EXP / 'results/smoke_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    require_curve()
    verify()
