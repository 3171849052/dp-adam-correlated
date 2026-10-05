"""Frozen two methods x three seeds, followed by sample-std reporting."""
from exp5.runtime import EXP
import json
import numpy as np
from exp5.config import FIXED, METHODS, FINAL_SEEDS, Trial
from exp5.launch_batch import launch, read_completed
from exp5.search import write_json


def report(rows, selected, root):
    methods = {}
    for method in METHODS:
        summaries = [r for r in rows if r['method'] == method]
        assert {r['seed'] for r in summaries} == set(FINAL_SEEDS)
        points = [dict(seed=r['seed'], final_top1=r['final']['test_top1'],
                       **r['diagnostics'], privacy=r['privacy']) for r in summaries]
        top1 = [p['final_top1'] for p in points]
        methods[method] = dict(selected_C=selected['methods'][method]['C'],
                        selected_lr=selected['methods'][method]['lr'], seeds=points,
                        mean_top1=float(np.mean(top1)), sample_std_top1=float(np.std(top1, ddof=1)),
                        best_seed=max(points, key=lambda p: (p['final_top1'], -p['seed']))['seed'],
                        worst_seed=min(points, key=lambda p: (p['final_top1'], p['seed']))['seed'],
                        workload='momentum', bandwidth=4 if method == METHODS[1] else 1,
                        mean_diagnostics={k: float(np.mean([p[k] for p in points])) for k in
                            ('clip_fraction', 'mean_unclipped_norm', 'mean_clip_factor',
                             'query_norm', 'noise_std', 'momentum_norm', 'update_norm')})
    result = dict(methods=methods, fixed=FIXED, std_ddof=1,
                  total_completed_search_trials=selected['total_completed_search_trials'])
    write_json(root / 'final_summary.json', result)
    lines = ['# Exp5 final three-seed comparison', '',
             '| method | C | lr | mean top1 | sample std | best seed | worst seed |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method, r in methods.items():
        lines.append(f"| {method} | {r['selected_C']} | {r['selected_lr']} | {r['mean_top1']:.6f} | {r['sample_std_top1']:.6f} | {r['best_seed']} | {r['worst_seed']} |")
    lines += ['', '| method | seed | final top1 | clip fraction | query norm | noise std | momentum norm |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for method, r in methods.items():
        for p in r['seeds']:
            lines.append(f"| {method} | {p['seed']} | {p['final_top1']:.6f} | {p['clip_fraction']:.6f} | {p['query_norm']:.6g} | {p['noise_std']:.6g} | {p['momentum_norm']:.6g} |")
    lines += ['', f"Completed search trials: {result['total_completed_search_trials']}",
              '', 'Privacy: replace-one; epsilon=8; delta=1e-5; five sparse direct participations.',
              'BandInvMF: momentum workload, beta=0.9, bandwidth=4. IID: identity temporal filter.',
              '', 'Full per-seed diagnostics and privacy/workload metadata: final_summary.json.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    return result


def run():
    selected = json.loads((EXP / 'results/search/selected_configs.json').read_text())
    assert selected['fixed'] == FIXED and set(selected['methods']) == set(METHODS)
    root = EXP / 'results/final'
    jobs = []
    for method in METHODS:
        cfg = selected['methods'][method]
        assert cfg['frozen']
        for seed in FINAL_SEEDS:
            trial = Trial(method, seed, cfg['lr'], cfg['C'])
            jobs.append(dict(trial=trial.asdict(), result_dir=str(root / trial.identity)))
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'frozen_configs.json', selected)
    launch(jobs)
    return report([read_completed(Trial(**j['trial']), j['result_dir']) for j in jobs], selected, root)


if __name__ == '__main__':
    run()
