"""Accuracy-only selection and paired, t-based final statistics."""
import csv
import json
import math
import numpy as np
from scipy.stats import t
from exp7b import BASE
from exp7b.config import *
from exp7b.audit import audit_trial, audit_pairing

def rows(category):
    return [json.loads(p.read_text()) for p in sorted((BASE / f'results/{category}/trials').glob('*/summary.json'))]

def best(candidates):
    completed = [r for r in candidates if r['status'] == 'completed' and not r['smoke']
                 and math.isfinite(r['final_test_top1'])]
    assert completed, 'No finite completed epoch-5 candidate'
    # Stable tie break by canonical identity, never by a diagnostic statistic.
    return max(completed, key=lambda r: (r['final_test_top1'], trial_id({k: r[k] for k in ('method','seed','lr','C','eps_scale')})))

def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)

def search_report():
    records = rows('search')
    if not records:
        return
    fields = ('method','seed','lr','C','eps_scale','status','final_test_top1','source','result_dir')
    write_csv(BASE / 'results/search/search_summary.csv', [{k: r[k] for k in fields} for r in records])
    save_json(BASE / 'results/search/search_summary.json', records)
    lines = ['# Search results', '', 'Selection objective: seed 20261001 epoch-5 final_test_top1 only.', '',
             '| Method | LR | C | eps_scale | Top-1 | Source |', '|---|---:|---:|---:|---:|---|']
    for method in SEARCH_METHODS:
        candidates = [r for r in records if r['method'] == method and r['status'] == 'completed']
        if candidates:
            r = best(candidates)
            lines.append(f"| {method} | {r['lr']} | {r['C']} | {r['eps_scale']} | {r['final_test_top1']:.4f} | {r['source']} |")
    (BASE / 'results/search/search_report.md').write_text('\n'.join(lines) + '\n')

def statistics(values):
    a = np.asarray(values, dtype=float)
    assert len(a) == 10 and np.isfinite(a).all()
    mean, std = float(a.mean()), float(a.std(ddof=1))
    se = std / math.sqrt(10)
    radius = float(t.ppf(.975, 9)) * se
    return dict(raw_top1=a.tolist(), mean=mean, sample_std=std, standard_error=se,
                ci95=[mean-radius, mean+radius], n=10)

def final_report():
    from exp7b.frozen import load_frozen
    frozen = load_frozen()
    records = rows('final')
    assert len(records) == 70
    assert {(r['method'], r['seed']) for r in records} == {(m, s) for m in METHODS for s in FINAL_SEEDS}
    for r in records:
        audit_trial(BASE.parent / r['result_dir'])
        assert all(r[k] == frozen[r['method']][k] for k in ('lr','C','eps_scale'))
    audit_pairing(records)
    lookup = {(r['method'], r['seed']): r['final_test_top1'] for r in records}
    summary = {m: statistics([lookup[m,s] for s in FINAL_SEEDS]) for m in METHODS}
    best_method = max(METHODS, key=lambda m: summary[m]['mean'])
    comparisons = [(SGD,IID),(MOMENTUM,SGD),(BIAS,MOMENTUM),(SGD_SCALE,SGD),
                   (MOMENTUM_SCALE,MOMENTUM),(BIAS_SCALE,BIAS),(BIAS_SCALE,MOMENTUM_SCALE),(best_method,IID)]
    effects = []
    for i, (a,b) in enumerate(comparisons):
        differences = [lookup[a,s] - lookup[b,s] for s in FINAL_SEEDS]
        st = statistics(differences)
        effects.append(dict(comparison='best method - IID' if i == 7 else f'{a} - {b}',
                            method_a=a, method_b=b, mean_difference=st['mean'], sample_std=st['sample_std'],
                            standard_error=st['standard_error'], ci95_low=st['ci95'][0], ci95_high=st['ci95'][1],
                            wins=sum(d > 0 for d in differences), n=10, differences=differences))
    fields = ('method','seed','lr','C','eps_scale','final_test_top1','result_dir')
    write_csv(BASE / 'results/final_multiseed.csv', [{k:r[k] for k in fields} for r in sorted(records, key=lambda r:(r['seed'], list(METHODS).index(r['method'])))])
    save_json(BASE / 'results/method_summary.json', summary)
    save_json(BASE / 'results/paired_effects.json', effects)
    write_csv(BASE / 'results/paired_effects.csv', [{k:v for k,v in e.items() if k != 'differences'} for e in effects])
    lines = ['# Exp7b final report', '', '70 full runs, 10 shared seeds; hyperparameters frozen before final.',
             'Accuracy and differences are fractions. CI uses Student t, df=9. Wins exclude ties.',
             'The best method comparison is descriptive and selected by final mean; it does not change hyperparameters.', '',
             '| Method | Mean | Sample std | SE | 95% CI |', '|---|---:|---:|---:|---|']
    for m, st in summary.items():
        lines += [f"| {m} | {st['mean']:.5f} | {st['sample_std']:.5f} | {st['standard_error']:.5f} | {st['ci95']} |"]
    lines += ['', '## Raw Top-1 in seed order 20261011–20261020', '']
    lines += [f"- {m}: {st['raw_top1']}" for m,st in summary.items()]
    lines += ['', '## Paired differences', '', '| Comparison | Mean difference | Sample std | 95% CI | Wins / 10 |', '|---|---:|---:|---|---:|']
    lines += [f"| {e['comparison']} | {e['mean_difference']:.5f} | {e['sample_std']:.5f} | [{e['ci95_low']:.5f}, {e['ci95_high']:.5f}] | {e['wins']}/10 |" for e in effects]
    lines += ['', '## Three workloads × two geometries', '',
              'Each cell uses its own frozen hyperparameters. Gains compare paired seeds.', '',
              '| Workload | Standard mean | Scale mean | Paired scale gain | Gain 95% CI | Wins / 10 |',
              '|---|---:|---:|---:|---|---:|']
    structure = {}
    for label, m in [('SGD',SGD),('Momentum',MOMENTUM),('Momentum-Bias',BIAS)]:
        effect = next(e for e in effects if (e['method_a'],e['method_b']) == (m+'-scale',m))
        lines += [f"| {label} | {summary[m]['mean']:.5f} | {summary[m+'-scale']['mean']:.5f} | {effect['mean_difference']:.5f} | [{effect['ci95_low']:.5f}, {effect['ci95_high']:.5f}] | {effect['wins']}/10 |"]
        structure[label] = dict(standard=dict(method=m,settings=frozen[m],statistics=summary[m]),
                                scale=dict(method=m+'-scale',settings=frozen[m+'-scale'],statistics=summary[m+'-scale']),
                                paired_scale_gain=effect)
    lines += ['', '| Workload | Standard (lr, C) | Scale (lr, C, eps_scale) |', '|---|---|---|']
    for label,m in [('SGD',SGD),('Momentum',MOMENTUM),('Momentum-Bias',BIAS)]:
        a,b = frozen[m],frozen[m+'-scale']
        lines += [f"| {label} | ({a['lr']}, {a['C']}) | ({b['lr']}, {b['C']}, {b['eps_scale']}) |"]
    from exp7b.bandinvmf import build_matrices, workload_error
    from exp7b.audit import array_hash
    from exp2.privacy import fixed_epoch_sensitivity
    lines += ['', '| Workload | Four noising coefficients d | Sensitivity of S | Full-workload error |', '|---|---|---:|---:|']
    for label,m in [('SGD',SGD),('Momentum',MOMENTUM),('Momentum-Bias',BIAS)]:
        d,S,W = build_matrices(cell_for(m)['noise'],250,4,.9)
        sensitivity = fixed_epoch_sensitivity(S,5,50)
        error = workload_error(W,d)
        structure[label]['matrix_design'] = dict(noising_coefficients=d.tolist(),
                    strategy_sensitivity=sensitivity,mean_squared_workload_error=error,
                    error_definition='||W D||_F^2 / 250',workload_sha256=array_hash(W),strategy_sha256=array_hash(S),
                    num_bands=4,k=5,spacing=50,full_workload_shape=[250,250])
        lines += [f"| {label} | {[round(float(x),9) for x in d]} | {sensitivity:.9f} | {error:.9f} |"]
    save_json(BASE / 'results/bandinvmf_structure.json', structure)
    lines += ['', 'Standard clips/noises gradients; Scale uses previous-vhat scaled query coordinates then inverse-scales before Adam.',
              'Within a workload, Standard and Scale share D and S=D^(-1). The Gaussian innovation standard deviation also depends on the frozen clipping C.',
              'The error column is ||W D||_F^2 / 250 for each stated workload. Momentum-Bias uses the full non-Toeplitz bias-corrected first-moment workload.',
              'All methods use add/remove zero-out adjacency, k=5, spacing=50, no sampling amplification, ε=8, δ=1e-5. Adam eps=1e-8 is separate from eps_scale.', '',
              '## Frozen settings', '', '```json', json.dumps(frozen, indent=2), '```']
    (BASE / 'results/final_report.md').write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    final_report() if args.final else search_report()
