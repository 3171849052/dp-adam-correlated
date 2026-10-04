"""Validate the saved real-data smoke artifacts for all five methods."""
import json
import hashlib
import numpy as np
import torch
import yaml
from exp3.bandinvmf import build_matrices
from exp3.spec import EXP3, METHODS, TrialSpec, read_completed


def main():
    specs = []
    for name in ('mf', 'baseline'):
        specs.extend(TrialSpec(**s) for s in json.loads((EXP3 / f'specs/smoke_{name}.json').read_text()))
    summaries = {s.method: read_completed(s) for s in specs}
    assert set(summaries) == set(METHODS)
    keys = ('initialization_sha256', 'classifier_initialization_sha256', 'checkpoint_sha256',
            'train_order_sha256', 'augmentation_trace_sha256')
    reference = summaries['mf_muon_standard']
    matrices = []
    for spec in specs:
        s = summaries[spec.method]
        assert s['optimizer_steps'] == 2 and s['physical_batches'] == 8
        assert s['noise_steps'] == (0 if spec.method == 'nonprivate_hybrid' else 2)
        assert len(s['epoch_test_top1']) == 1 and 0 <= s['final_test_top1'] <= 1
        assert all(s[k] == reference[k] for k in keys)
        order = np.load(spec.directory / 'train_order.npy')
        assert order.shape == (50000,) and len(np.unique(order)) == 50000
        assert hashlib.sha256(order.tobytes()).hexdigest() == s['train_order_sha256']
        cfg = yaml.safe_load((spec.directory / 'config.yaml').read_text())
        assert cfg['download'] is False and cfg['visible_devices'] in ('1', '2', '3')
        assert s['diagnostics']['probe_count'] == 2 and len(s['diagnostics']['records']) == 6
        assert {r['step'] for r in s['diagnostics']['records']} == {1,2}
        assert all(np.isfinite(r['update_gain_mean']) and np.isfinite(r['phi_gain_std']) for r in s['diagnostics']['records'])
        for row in s['diagnostics']['records']:
            assert np.isclose(row['update_gain_mean'], row['shape_factor'] * row['phi_gain_mean'])
            if spec.method == 'nonprivate_hybrid':
                assert row['noise_weighted_update_gain_mean'] is None
            else:
                assert np.isclose(row['noise_weighted_update_gain_mean'], s['innovation_std_sum']/1000*row['update_gain_mean'])
        saved = torch.load(spec.directory / 'final.pt', map_location='cpu', weights_only=True)
        assert saved['logical_steps'] == 2
        assert all(torch.isfinite(v).all() for v in saved['model'].values())
        pre_ns = [state['pre_ns'] for state in saved['optimizer']['state'].values() if 'pre_ns' in state]
        assert len(pre_ns) == 48 and all(torch.isfinite(h).all() for h in pre_ns)
        kind = 'momentum_bandinvmf' if spec.method.startswith('mf_') else 'iid'
        expected = build_matrices(kind, 250, 4, .9)
        with np.load(spec.directory / 'matrices.npz') as data:
            for key, value in zip(('noising_coefficients', 'strategy', 'workload_coefficients'), expected):
                np.testing.assert_array_equal(data[key], value)
        if spec.method.startswith('mf_'):
            trajectory=torch.load(spec.directory/'muon_trajectory.pt',map_location='cpu',weights_only=True)
            assert len(trajectory)==3 and all(len(frames)==2 for frames in trajectory.values())
            assert s['frozen_trajectory_muon_mf']['steps']==2
            with np.load(spec.directory / 'matrices.npz') as data:
                matrices.append({k: data[k].copy() for k in data.files})
    for data in matrices[1:]:
        for key in matrices[0]:
            np.testing.assert_array_equal(data[key], matrices[0][key])
    # Same first step: identical initial state, identity geometry, coefficients and noise RNG.
    first = [summaries[m]['diagnostics']['records'][:3] for m in METHODS[2:]]
    assert first[0] == first[1] == first[2]
    assert all(summaries[m]['frozen_trajectory_muon_mf']['temporal_probe_sha256'] ==
               reference['frozen_trajectory_muon_mf']['temporal_probe_sha256'] for m in METHODS[2:])
    result = dict(status='passed', methods=5, logical_steps_per_trial=2, physical_batches_per_trial=8,
                  paired_initialization_order_augmentation=True, mf_matrices_identical=True,
                  finite_saved_models=True, saved_pre_ns_matrices_per_trial=48,
                  diagnostics_records_per_trial=6,
                  final_test_top1={m:summaries[m]['final_test_top1'] for m in METHODS},
                  optimizer_steps={m:summaries[m]['optimizer_steps'] for m in METHODS},
                  noise_steps={m:summaries[m]['noise_steps'] for m in METHODS})
    (EXP3 / 'results/smoke_v2_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
