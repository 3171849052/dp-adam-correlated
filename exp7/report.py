"""Accuracy-only selection and standardized live/exported result summaries."""
import csv
import json
from exp7 import BASE
from exp7.config import METHODS, SCALE, trial_id, save_json

def rows():
    return [json.loads(p.read_text()) for p in sorted((BASE / 'results/trials').glob('*/summary.json'))]

def best(candidates):
    completed = [r for r in candidates if r['status'] == 'completed' and not r['smoke']]
    if not completed:
        raise RuntimeError('No completed accuracy candidates')
    return max(completed, key=lambda r: (r['final_test_top1'], -r['C'], -r['lr']))

def export(reasons=None, complete=False, stage='running'):
    all_rows = rows()
    keys = ['method', 'seed', 'lr', 'C', 'eps_scale', 'K', 'status', 'final_test_top1', 'source', 'result_dir', 'historical_result_dir', 'error']
    compact = [{k: r.get(k) for k in keys} for r in all_rows]
    for r in compact:
        r['K'] = None if r['eps_scale'] is None else r['C'] * r['eps_scale']
    save_json(BASE / 'results/search_summary.json', compact)
    with (BASE / 'results/search_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(compact)
    selected = {}
    for method in METHODS:
        choices = [r for r in all_rows if r['method'] == method and r['status'] == 'completed']
        if not choices:
            continue
        winner = best(choices)
        selected[method] = {k: winner[k] for k in ('seed', 'lr', 'C', 'eps_scale', 'final_test_top1', 'source', 'result_dir')}
        selected[method].update(num_bands=None if method == 'dp-adam-iid' else 4,
                                utility=winner['final_test_top1'], stop_reason=(reasons or {}).get(method, 'search in progress'))
    save_json(BASE / 'results/selected_configs.json', selected)
    counts = dict(new_trials=sum(r['source'] == 'new' for r in all_rows),
                  new_completed=sum(r['source'] == 'new' and r['status'] == 'completed' for r in all_rows),
                  historical_reused=sum(r['source'] == 'historical' for r in all_rows),
                  numerical_failures=sum(r['status'] == 'numerical_failure' for r in all_rows))
    save_json(BASE / 'results/search_state.json', dict(status='completed' if complete else 'running',
              stage=stage, seed=20261001, final_seeds_run=False, counts=counts, stop_reasons=reasons or {}))
    lines = ['# Exp7 search-seed report', '', f'Status: {"completed" if complete else "running"}; stage: {stage}.',
             'Seed: 20261001. Objective: epoch-5 final_test_top1 only. No final multi-seed runs.',
             f'New search trials: {counts["new_trials"]}; historical reused: {counts["historical_reused"]}; numerical failures: {counts["numerical_failures"]}.',
             'Smoke trials are isolated and excluded from selection.', '',
             '| Method | LR | C | eps_scale | Top1 | Source | Stop reason |', '|---|---:|---:|---:|---:|---|---|']
    for method, r in selected.items():
        lines.append(f'| {method} | {r["lr"]:g} | {r["C"]:g} | {r["eps_scale"]} | {r["utility"]:.4f} | {r["source"]} | {r["stop_reason"]} |')
    by_id = {trial_id({k:r[k] for k in ('method','seed','lr','C','eps_scale')}):r for r in all_rows}
    lines += ['', '| Scale stage | Best candidate LR | C | eps_scale | Top1 |', '|---|---:|---:|---:|---:|']
    for stage_name in ('coarse', 'scale_joint', 'scale_lr', 'scale_lr_extension', 'scale_refinement'):
        path = BASE / f'results/{stage_name}_candidates.json'
        if not path.exists():
            continue
        candidates = [by_id[trial_id(j)] for j in json.loads(path.read_text())
                      if j['method'] == SCALE and trial_id(j) in by_id and by_id[trial_id(j)]['status'] == 'completed']
        if candidates:
            r = best(candidates)
            lines.append(f"| {stage_name} | {r['lr']:g} | {r['C']:g} | {r['eps_scale']:g} | {r['final_test_top1']:.4f} |")
    lines += ['', 'All candidates (including numerical failures) are in search_summary.csv/json.',
              'Scale coarse coordinate is K=C*eps_scale; workload is SGD/prefix-sum, and scale uses previous completed vhat.',
              'Search is bounded at 28 new trials; one directional LR extension and one final local refinement are allowed.',
              'Historical evidence was audited against local checkpoint, initialization, training order, matrices, calibration and protocol; see history_audit.json.',
              'Runtime uses unchanged Exp2 mathematical kernels; full-size smoke verifies three GPU slots before search.', '']
    for r in all_rows:
        if r.get('checkpoint_note'):
            lines += [f"Checkpoint limitation ({r['result_dir']}): {r['checkpoint_note']}", '']
    (BASE / 'results/search_report.md').write_text('\n'.join(lines))
    return selected

if __name__ == '__main__':
    state_path = BASE / 'results/search_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    export(state.get('stop_reasons'), complete=state.get('status') == 'completed', stage=state.get('stage', 'running'))
    print('Wrote exp7/results/selected_configs.json, search_summary.csv/json, search_report.md')
