"""Paired cross-seed summaries. Holdouts are clustered by independently fitted seed."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from exp13.io import file_hash, local_path, save_json


def statistics(values):
    a = np.asarray(values, dtype=np.float64)
    n = len(a)
    assert n >= 2 and np.isfinite(a).all()
    mean, std = float(a.mean()), float(a.std(ddof=1))
    half = float(t.ppf(.975, n-1) * std / np.sqrt(n))
    return dict(n=n, mean=mean, sample_std=std, ci95_low=mean-half, ci95_high=mean+half,
                min=float(a.min()), max=float(a.max()))


def flatten(values):
    result = {k: v for k, v in values.items() if isinstance(v, (float, int))}
    result.update({f'd{i}': x for i, x in enumerate(values['d'])})
    for section in ('vhat_all', 'vhat_final'):
        result[f'{section}_mean'] = values[section]['mean']
        result[f'{section}_std'] = values[section]['std']
        for key, value in values[section]['quantiles'].items():
            result[f'{section}_{key}'] = value
    return result


def build(output):
    output = local_path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    p = manifest['protocol']
    rows, clustered, gaps = [], [], []
    for n in p['horizons']:
        for seed in p['seeds']:
            dest = output / f'T{n}' / f'seed{seed}'
            record = json.loads((dest / 'summary.json').read_text())
            assert (record['T'], record['seed'], record['K']) == (n, seed, p['coordinates'])
            for name, sha in record['artifacts'].items():
                assert file_hash(dest / name) == sha, dest / name
            for method in p['methods']:
                inside = flatten(record['in_sample'][method])
                rows.append(dict(T=n, fit_seed=seed, evaluation_seed=seed,
                                 split='in_sample', method=method, **inside))
                clustered.append(dict(T=n, fit_seed=seed, split='in_sample', method=method, **inside))
                tests = []
                for test in record['holdout']:
                    v = flatten(test['methods'][method])
                    tests.append(v)
                    rows.append(dict(T=n, fit_seed=seed, evaluation_seed=test['seed'],
                                     split='holdout', method=method, **v))
                assert len(tests) == p['holdouts_per_fit']
                average = {key: float(np.mean([v[key] for v in tests])) for key in tests[0]}
                clustered.append(dict(T=n, fit_seed=seed, split='holdout', method=method, **average))
                gap = {key: inside[key]-average[key] for key in average if '_improvement_vs_' in key}
                gaps.append(dict(T=n, fit_seed=seed, method=method, **gap))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    stats, gap_stats = [], []
    for n in p['horizons']:
        for method in p['methods']:
            for split in ('in_sample', 'holdout'):
                group = [r for r in clustered if (r['T'], r['method'], r['split']) == (n, method, split)]
                metrics = [key for key in group[0] if key not in ('T', 'fit_seed', 'split', 'method')]
                for metric in metrics:
                    stats.append(dict(T=n, method=method, split=split, metric=metric,
                                      **statistics([r[metric] for r in group])))
            group = [r for r in gaps if (r['T'], r['method']) == (n, method)]
            for metric in group[0]:
                if '_improvement_vs_' in metric:
                    gap_stats.append(dict(T=n, method=method, metric=metric,
                                          **statistics([r[metric] for r in group])))
    # Paired differences: actual-noise fit minus each comparator, within fit seed.
    paired = []
    for n in p['horizons']:
        for split in ('in_sample', 'holdout'):
            for baseline in p['methods'][:-1]:
                for metric in ('J', 'J_real'):
                    differences = []
                    for seed in p['seeds']:
                        lookup = {r['method']: r for r in clustered
                                  if (r['T'], r['split'], r['fit_seed']) == (n, split, seed)}
                        differences.append(lookup['actual_self_consistent'][metric]-lookup[baseline][metric])
                    paired.append(dict(T=n, split=split, baseline=baseline, metric=metric,
                                       difference='actual minus baseline', **statistics(differences)))
    save_json(output / 'statistics.json', stats)
    save_json(output / 'paired_differences.json', paired)
    save_json(output / 'overfit_gaps.json', gap_stats)
    text = [f'# Exp13 {p["kind"]} numerical results', '',
            f'K={p["coordinates"]}; {len(p["seeds"])} independently fitted seeds; '
            f'{p["holdouts_per_fit"]} independent holdouts per fit. Float64, no model execution.', '',
            'J is the coordinate-averaged Frobenius workload objective with endogenous vhat. '
            'J_real is the measured mean squared cumulative pure-noise Adam update (learning rate 1). '
            'They are different functionals. All rows below evaluate actual N² denominators, '
            'including the matrix fitted by the expected-v comparator.', '',
            'Holdout results average evaluations within each fitted seed before cross-seed statistics; '
            '95% CIs use the Student t distribution. Relative improvements are paired percentages.', '',
            '| T | split | method | J mean ± std | J_real mean ± std | '
            'J improvement vs IID (%) | J_real improvement vs IID (%) |',
            '|---|---|---|---:|---:|---:|---:|']
    for n in p['horizons']:
        for split in ('in_sample', 'holdout'):
            for method in p['methods']:
                s = {r['metric']: r for r in stats if (r['T'], r['split'], r['method']) == (n, split, method)}
                text.append(f'| {n} | {split} | {method} | '
                            f'{s["J"]["mean"]:.6g} ± {s["J"]["sample_std"]:.3g} | '
                            f'{s["J_real"]["mean"]:.6g} ± {s["J_real"]["sample_std"]:.3g} | '
                            f'{s["J_improvement_vs_iid_pct"]["mean"]:.3f} | '
                            f'{s["J_real_improvement_vs_iid_pct"]["mean"]:.3f} |')
    text += ['', 'Coefficients, noise calibration, inverse amplification, vhat distributions, '
             'Exp9-relative improvements and all CIs are in metrics.csv/statistics.json. '
             'overfit_gaps.json reports in-sample minus holdout improvement: positive values '
             'indicate a loss of improvement on fresh innovations. paired_differences.json '
             'compares actual-noise fitting with all three comparators.']
    (output / 'report.md').write_text('\n'.join(text)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    build(parser.parse_args().output)
