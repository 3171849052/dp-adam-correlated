"""Independently replay persisted smoke matrices with explicit coordinate workloads."""
import argparse
import json
from pathlib import Path
import numpy as np
from exp13 import BASE
from exp13.io import file_hash, local_path, save_json
from exp13.math import BETA1, BETA2, EPS, digest


def verify(output):
    output = local_path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['protocol']['kind'] == 'smoke'
    verified = []
    for path in sorted(output.glob('T*/seed*/summary.json')):
        record = json.loads(path.read_text())
        xi = np.load(path.parent / 'xi.npy', allow_pickle=False)
        assert digest(xi) == record['xi_sha256']
        n, k = xi.shape
        L = np.tril(np.ones((n, n)))
        M = np.zeros((n, n))
        for t in range(n):
            for s in range(t+1):
                M[t, s] = (1-BETA1)*BETA1**(t-s)/(1-BETA1**(t+1))
        checked = {}
        for method, values in record['in_sample'].items():
            with np.load(path.parent / f'{method}_matrices.npz') as arrays:
                D, C, Q, sigma = [arrays[key] for key in ('D', 'C', 'Q', 'sigma')]
            N = sigma * D @ xi
            state = np.zeros(k)
            v = []
            for t, row in enumerate(N):
                state = BETA2*state + (1-BETA2)*row**2
                v.append(state/(1-BETA2**(t+1)))
            a = 1/(np.sqrt(np.stack(v))+EPS)
            W = [L @ np.diag(a[:, c]) @ M for c in range(k)]
            direct_J = float(sigma**2*sum(np.square(w @ D).sum() for w in W)/(n*k))
            direct_real = float(np.mean(np.cumsum((M @ N)*a, axis=0)**2))
            np.testing.assert_allclose(direct_J, values['J'], rtol=2e-12)
            np.testing.assert_allclose(direct_real, values['J_real'], rtol=2e-12)
            explicit_Q = np.zeros((4, 4))
            for w in W:
                shifts = [np.pad(w[:, i:], ((0, 0), (0, i))) for i in range(4)]
                for i in range(4):
                    for j in range(4):
                        explicit_Q[i, j] += np.sum(shifts[i]*shifts[j])/(n*k)
            np.testing.assert_allclose(Q, explicit_Q, rtol=2e-12)
            # Absolute residual scales with C near Exp9's 1e8 boundary.
            backward_error = np.linalg.norm(C @ D-np.eye(n), ord=np.inf) / (
                np.linalg.norm(C, ord=np.inf)*np.linalg.norm(D, ord=np.inf))
            assert backward_error <= 10*n*np.finfo(np.float64).eps
            assert np.abs(C).max() <= 1e8
            checked[method] = dict(J=direct_J, J_real=direct_real,
                                   inverse_backward_error=float(backward_error))
        for method in ('expected_self_consistent', 'actual_self_consistent'):
            trace = [json.loads(line) for line in
                     (path.parent / f'{method}_trajectory.jsonl').read_text().splitlines()]
            fit = record['fits'][method]
            assert len(trace) == fit['evaluations']
            assert all(np.isfinite(row['J']) and row['J'] > 0 for row in trace)
            assert fit['objective_at_solution'] == min(r['objective'] for r in fit['starts'])
        assert set(h['seed'] for h in record['holdout']).isdisjoint(manifest['protocol']['seeds'])
        verified.append(dict(seed=record['seed'], physical_gpu=record['physical_gpu'], methods=checked))
    assert len(verified) == 4 and {r['physical_gpu'] for r in verified} == {0, 1, 2, 3}
    stats = json.loads((output / 'statistics.json').read_text())
    assert all(s['n'] == 4 for s in stats)
    for name, sha in manifest['source_sha256'].items():
        assert file_hash(BASE / name) == sha, name
    save_json(output / 'verification.json', dict(verified=verified,
              explicit_workload_and_Q_replay=True, actual_adam_recurrence_replay=True,
              source_hashes_unchanged=True, clustered_statistics_n=4))
    print('Verified 16 persisted matrices: explicit coordinate Workload/Q, Adam recursion, '
          'trajectory counts, four GPU jobs, and clustered statistics.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=BASE / 'results/smoke')
    verify(parser.parse_args().output)
