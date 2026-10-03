"""Explicitly run the fifteen paired trials after configurations are frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from exp3.launch_batch import run_queue
from exp3.spec import METHODS, TrialSpec, EXP3

FINAL_SEEDS = (20261011, 20261012, 20261013)


def final_specs(frozen, directory):
    assert set(frozen) == set(METHODS)
    specs = []
    for seed in FINAL_SEEDS:
        for method in METHODS:
            parameters = frozen[method]
            assert {'muon_lr', 'adam_lr', 'max_grad_norm'} <= set(parameters)
            specs.append(TrialSpec(method=method, seed=seed, **parameters,
                                   result_dir=str(directory / f'{method}/seed_{seed}')))
    assert len(specs) == 15 and all(not s.smoke for s in specs)
    return specs


def verify_pairing(rows):
    for seed in FINAL_SEEDS:
        paired = [row for row in rows if row['spec']['seed'] == seed]
        assert {r['spec']['method'] for r in paired} == set(METHODS)
        reference = paired[0]
        for row in paired[1:]:
            for key in ('checkpoint_sha256', 'initialization_sha256', 'classifier_initialization_sha256',
                        'train_order_sha256', 'augmentation_trace_sha256'):
                assert row['summary'][key] == reference['summary'][key], f'Pairing mismatch: {seed} {key}'
            np.testing.assert_array_equal(np.load(TrialSpec(**row['spec']).directory / 'train_order.npy'),
                                          np.load(TrialSpec(**reference['spec']).directory / 'train_order.npy'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-config', required=True, type=Path)
    parser.add_argument('--result-dir', required=True, type=Path)
    args = parser.parse_args()
    directory = args.result_dir.resolve()
    assert directory.is_relative_to(EXP3) and directory != EXP3
    contents = args.frozen_config.read_bytes()
    frozen = json.loads(contents)
    specs = final_specs(frozen, directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / 'frozen_config.json'
    if snapshot.exists():
        assert snapshot.read_bytes() == contents, 'Frozen configuration changed'
    else:
        snapshot.write_bytes(contents)
    summary = run_queue(specs, directory)
    if summary['status'] != 'completed':
        raise SystemExit(1)
    verify_pairing(summary['trials'])
    aggregate = {}
    for method in METHODS:
        per_seed = {str(r['spec']['seed']): r['summary']['final_test_top1']
                    for r in summary['trials'] if r['spec']['method'] == method}
        values = [per_seed[str(seed)] for seed in FINAL_SEEDS]
        aggregate[method] = dict(final_test_top1_mean=float(np.mean(values)),
                                 final_test_top1_sample_std=float(np.std(values, ddof=1)), per_seed=per_seed)
    final = dict(status='completed', paired=True, seeds=list(FINAL_SEEDS), trials=15,
                 frozen_config_sha256=hashlib.sha256(contents).hexdigest(), methods=aggregate)
    (directory / 'final_summary.json').write_text(json.dumps(final, indent=2))
    print(json.dumps(final))


if __name__ == '__main__':
    main()
