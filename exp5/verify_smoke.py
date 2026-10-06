"""Audit the three real GPU smoke runs and their geometry/performance."""
from exp5.runtime import EXP
import json
import numpy as np
from exp5.config import FIXED, METHODS
from exp5.train import write_json


def verify():
    summaries = []
    devices = set()
    for folder in sorted((EXP / 'results/smoke').iterdir()):
        s = json.loads((folder / 'summary.json').read_text())
        c = json.loads((folder / 'config.json').read_text())
        assert s['status'] == 'completed' and s['smoke'] and s['fixed'] == FIXED
        assert s['optimizer_steps'] == s['noise_steps'] == 1 and s['physical_batches'] == 10
        assert s['privacy']['per_step_sensitivity'] == .002
        assert s['privacy']['participation'] == 'fixed_epoch_sparse'
        assert s['privacy']['workload'] == 'momentum' and s['privacy']['direct_participations'] == 5
        assert all(np.isfinite(v) for v in s['diagnostics'].values())
        if s['method'] == METHODS[0]:
            assert np.isclose(s['privacy']['innovation_std'], .002*np.sqrt(5)/s['privacy']['mu'])
        else:
            assert s['privacy']['num_bands'] == 4
        devices.add(c['visible_devices'])
        summaries.append(s)
    assert len(summaries) == 3 and devices == {'0', '1', '2'}
    paired = [s for s in summaries if s['seed'] == 20261001]
    assert len({s['initialization_sha256'] for s in paired}) == 1
    for k in ('query_norm', 'clip_fraction', 'mean_transformed_sample_norm',
              'raw_mean_gradient_norm', 'mean_clip_factor', 'batch_coherence'):
        assert np.isclose(paired[0]['diagnostics'][k], paired[1]['diagnostics'][k], rtol=1e-6)
    result = dict(status='passed', gpus=[0, 1, 2], paired_initialization_and_query=True,
                  trials=[dict(method=s['method'], seed=s['seed'], privacy=s['privacy'],
                               diagnostics=s['diagnostics'], peak_allocated_mib=s['peak_allocated_mib'],
                               peak_reserved_mib=s['peak_reserved_mib']) for s in summaries])
    write_json(EXP / 'results/smoke_verification.json', result)
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    verify()
