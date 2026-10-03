"""Check six real one-step trials and their paired saved artifacts."""
import json
import numpy as np
import torch
from exp2.bandinvmf import build_matrices, materialize
from exp2.cells import CELLS, FINAL_SEEDS
from exp2.diagnostics import save_json
from exp2.sweep import verify_pairing
from exp2.train import ROOT


def main():
    base = ROOT / 'exp2/results'
    results = []
    for method, cell in CELLS.items():
        directory = base / 'smoke' / method / f'seed_{FINAL_SEEDS[0]}'
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['smoke'] and summary['status'] == 'completed'
        assert summary['optimizer_steps'] == summary['noise_steps'] == 1
        assert summary['physical_batches'] == 4
        assert summary['final']['train_examples'] == 1000
        assert summary['planned_total_steps'] == 250
        d, c, w = build_matrices(cell['noise'], 250, 4, .9)
        with np.load(directory / 'matrices.npz') as matrices:
            np.testing.assert_array_equal(matrices['noising_coefficients'], d)
            np.testing.assert_array_equal(matrices['strategy'], c)
            np.testing.assert_array_equal(matrices['W'], materialize(w, 250))
            np.testing.assert_array_equal(matrices['M'], materialize(d, 250) *
                                          summary['calibration']['innovation_std_sum'])
        checkpoint = torch.load(directory / 'final.pt', map_location='cpu', weights_only=True)
        assert checkpoint['logical_steps'] == 1
        state = checkpoint['optimizer']['state']
        parameters = checkpoint['optimizer']['param_groups'][0]['params']
        vhat = np.concatenate([(state[index]['exp_avg_sq'] / (1 - .999)).numpy().ravel()
                               for index in parameters])
        assert all(int(state[index]['step']) == 1 for index in parameters)
        with np.load(directory / 'mechanism_trace.npz') as trace:
            indices = trace['coordinate_indices']
            assert trace['p_trace'].shape == trace['r_trace'].shape == trace['s_trace'].shape == (1, 2048)
            scale = 10. if cell['geometry'] == 'scale' else 1.
            np.testing.assert_array_equal(trace['s_trace'], np.full((1, 2048), scale))
            expected_p = 1 / (np.sqrt(vhat[indices]) + 1e-8)
            expected_r = expected_p / scale
            np.testing.assert_allclose(trace['p_trace'][0], expected_p, rtol=2e-6)
            np.testing.assert_allclose(trace['r_trace'][0], expected_r, rtol=2e-6)
            np.testing.assert_allclose(trace['r_trace'], trace['p_trace'] / trace['s_trace'], rtol=1e-6)
            for key in ('p_trace', 's_trace', 'r_trace'):
                assert np.isfinite(trace[key]).all()
        results.append((dict(method=method, seed=FINAL_SEEDS[0]), directory, summary))
    verify_pairing(results)
    save_json(base / 'smoke_verification.json', dict(status='passed', trials=6,
        logical_steps_per_trial=1, train_examples_per_trial=1000,
        checkpoint_sha256=results[0][2]['pretrained_checkpoint_sha256'],
        checks=['one noise and Adam output per 4 physical batches', 'saved current p/s/r matches completed Adam state',
                'previous vhat_0 gives Scale s=10', 'actual M/W match noise coefficients and workload',
                'paired backbone/classifier/model/order/rolling augmentation hash/coordinates']))
    print('PASS: 6 smoke trials; physical 250 × 4; one noise/Adam step; paired artifacts and p/s/r/M/W verified.')


if __name__ == '__main__':
    main()
