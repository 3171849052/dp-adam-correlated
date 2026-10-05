"""Audit completed real GPU smoke trials against the fixed privacy protocol."""
from exp5.runtime import EXP
import json
import numpy as np
from exp5.config import FIXED, METHODS
from exp5.search import write_json


def verify():
    folders = sorted((EXP / 'results/smoke').iterdir())
    assert len(folders) == 3
    summaries, configs = [], []
    for folder in folders:
        s = json.loads((folder / 'summary.json').read_text())
        c = json.loads((folder / 'config.json').read_text())
        assert s['status'] == 'completed' and s['smoke']
        assert s['fixed'] == FIXED
        assert s['optimizer_steps'] == s['noise_steps'] == 1
        assert s['physical_batches'] == 10
        assert s['privacy']['per_query_sensitivity'] == 2 * s['C'] / 1000
        assert s['privacy']['participation'] == 'fixed_epoch_sparse'
        assert s['privacy']['direct_participations'] == 5
        assert s['privacy']['workload'] == 'momentum'
        assert all(np.isfinite(v) for v in s['final'].values() if isinstance(v, (float, int)))
        assert all(np.isfinite(v) for v in s['diagnostics'].values())
        if s['method'] == METHODS[0]:
            assert np.isclose(s['privacy']['innovation_std'], 2 * s['C'] * np.sqrt(5) / (1000 * s['privacy']['mu']))
        else:
            assert s['privacy']['num_bands'] == 4
        summaries.append(s)
        configs.append(c)
    assert {c['visible_devices'] for c in configs} == {'0', '1', '2'}
    assert {s['method'] for s in summaries} == set(METHODS)
    paired = [s for s in summaries if s['seed'] == 20261001]
    assert len({s['initialization_sha256'] for s in paired}) == 1
    for k in ('query_norm', 'clip_fraction', 'mean_unclipped_norm', 'mean_clip_factor'):
        assert np.isclose(paired[0]['diagnostics'][k], paired[1]['diagnostics'][k], rtol=1e-6)
    result = dict(status='passed', trials=[dict(method=s['method'], seed=s['seed'],
                  final=s['final'], privacy=s['privacy'], wall_seconds=s['wall_seconds']) for s in summaries],
                  gpus=[0, 1, 2], paired_initialization_and_query=True)
    write_json(EXP / 'results/smoke_verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    verify()
