"""Two frozen methods x three final seeds, with ddof=1 reports."""
from exp5.runtime import EXP, output_path
import argparse
import json
import numpy as np
from exp5.config import FIXED, METHODS, FINAL_SEEDS, Trial
from exp5.launch_batch import launch, read_completed
from exp5.train import write_json, DIAGNOSTICS


def report(rows, selected, root):
    root = output_path(root)
    methods = {}
    for method in METHODS:
        summaries = [r for r in rows if r['method'] == method]
        assert {r['seed'] for r in summaries} == set(FINAL_SEEDS)
        points = [dict(seed=r['seed'], final_top1=r['final']['test_top1'],
                       train_loss=r['final']['train_loss'], train_top1=r['final']['train_top1'],
                       test_loss=r['final']['test_loss'], **r['diagnostics'], privacy=r['privacy']) for r in summaries]
        top1 = [p['final_top1'] for p in points]
        cfg = selected['methods'][method]
        methods[method] = dict(selected_tau=cfg['tau'], C=1., selected_lr=cfg['lr'], seeds=points,
                              mean_top1=float(np.mean(top1)), sample_std_top1=float(np.std(top1, ddof=1)),
                              workload='momentum', bandwidth=4 if method == METHODS[1] else 1,
                              mean_diagnostics={k: float(np.mean([p[k] for p in points])) for k in DIAGNOSTICS})
    iid, mf = METHODS
    result = dict(methods=methods, fixed=FIXED, std_ddof=1,
                  bandinvmf_minus_iid_top1=methods[mf]['mean_top1']-methods[iid]['mean_top1'],
                  total_completed_search_trials=selected['total_completed_search_trials'],
                  privacy_scope='Per-run fixed-config calibration; nonprivate probe, selection, diagnostics and combined releases are not covered')
    write_json(root / 'final_summary.json', result)
    lines = ['# Exp5 coordinate-normalized SGDM', '',
             '| method | tau | C | lr | mean top1 | sample std (ddof=1) |',
             '|---|---:|---:|---:|---:|---:|']
    for method, r in methods.items():
        lines.append(f"| {method} | {r['selected_tau']:.8g} | 1 | {r['selected_lr']:.8g} | {r['mean_top1']:.6f} | {r['sample_std_top1']:.6f} |")
    lines += ['', '| method | seed | final top1 | raw mean gradient norm | mean transformed norm | clip fraction | query norm | coherence | noise std | momentum norm |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, r in methods.items():
        for p in r['seeds']:
            lines.append(f"| {method} | {p['seed']} | {p['final_top1']:.6f} | {p['raw_mean_gradient_norm']:.6g} | {p['mean_transformed_sample_norm']:.6g} | {p['clip_fraction']:.6f} | {p['query_norm']:.6g} | {p['batch_coherence']:.6g} | {p['noise_std']:.6g} | {p['momentum_norm']:.6g} |")
    lines += ['', 'Trajectory diagnostic means over all 250 logical steps:', '',
              '| metric | IID | BandInvMF |', '|---|---:|---:|']
    for key in DIAGNOSTICS:
        a, b = methods[iid]['mean_diagnostics'][key], methods[mf]['mean_diagnostics'][key]
        lines.append(f'| {key} | {a:.8g} | {b:.8g} |')
    mu = rows[0]['privacy']['mu']
    lines += ['', f'Calibrated target mu: {mu:.12g}.',
              'transformed_norm_std is the mean within-logical-batch sample std; epoch CSVs also retain pooled epoch statistics.',
              'logical_step_seconds includes epoch loader startup; the Phase 1 report separately records steady-state timing.', '']
    lines += ['', f"BandInvMF minus IID mean top1: {result['bandinvmf_minus_iid_top1']:.6f}", '',
              'Replace-one step sensitivity=0.002; epsilon=8; delta=1e-5; five fixed sparse participations.',
              'BandInvMF uses the exp2 momentum workload, beta=0.9, bandwidth=4.',
              'Privacy calibration applies to a single run conditional on fixed configuration. The nonprivate scale probe, test-based selection, raw diagnostics, and combined releases are excluded.',
              'Full seed diagnostics and accountant metadata: final_summary.json.', '']
    output_path(root / 'report.md').write_text('\n'.join(lines))
    return result


def run():
    selected = json.loads((EXP / 'results/search/selected_configs.json').read_text())
    assert selected['fixed'] == FIXED and set(selected['methods']) == set(METHODS)
    root = EXP / 'results/final'
    jobs = []
    for method in METHODS:
        cfg = selected['methods'][method]
        assert cfg['frozen'] and cfg['C'] == 1.
        for seed in FINAL_SEEDS:
            trial = Trial(method, seed, cfg['lr'], cfg['tau'])
            jobs.append(dict(trial=trial.asdict(), result_dir=str(root / trial.identity)))
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'frozen_configs.json', selected)
    launch(jobs)
    return report([read_completed(Trial(**j['trial']), j['result_dir']) for j in jobs], selected, root)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1,2', choices=['0,1,2'])
    p.parse_args()
    run()
