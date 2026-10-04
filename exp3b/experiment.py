"""Continue utility tuning, freeze configs, run seeds and fixed-signal cancellation."""
import argparse
import csv
import json
import sys

from exp3b import BASE, ROOT
from exp3b.history import choose, key, read_inputs, utility_config
from exp3b.scheduler import run_commands, run_trials
from exp3b.spec import ADAM, HYBRID, PARAMETERS, RunSpec, write_json

FINAL_SEEDS = (20261011, 20261012, 20261013)


def protocol():
    values = json.loads((BASE / 'specs/protocol.json').read_text())
    assert values['gpus'] == [0, 1, 2, 3] and values['max_concurrent_trials'] == 4
    assert values['objective'] == 'final_test_top1'
    assert values['final_seeds'] == list(FINAL_SEEDS)
    assert values['scale_refinement_budget'] in (3, 4)
    assert values['standard_initial_lrs'] == [.009, .012]
    assert values['adam_standard']['initial_lrs'] == [.003, .007]
    assert values['adam_standard']['clips'] == [10., 30., 100.]
    assert values['adam_standard']['max_lr_extensions'] <= 4
    return values


def config(row):
    return {p: row[p] for p in PARAMETERS}


def selection(inputs, rows, frozen):
    choices = {m: choose(rows, m, m.endswith(('normscale', 'spectralscale'))) for m in HYBRID + ADAM}
    return dict(status='frozen' if frozen else 'pending_refinement', objective='final_test_top1',
                configs={m: utility_config(row) for m, row in choices.items()},
                provenance={m: dict(result_dir=r['result_dir'], final_test_top1=r['final_test_top1'],
                                    origin=r['origin'], source_method=r.get('source_method', m),
                                    source_csv=r.get('source_csv', 'exp3/results/search_history.json'))
                            for m, r in choices.items()},
                mf_adam_tuning=dict(status='completed' if frozen else 'pending_continuation',
                    objective='maximum final_test_top1',
                    history='all completed Exp2 momentum_standard / momentum_standard_matched utility observations',
                    selection='Exp3b utility continuation; matched label and clipping matching never rank configurations',
                    fixed_anchor_used=False, continuation='exp3b/results/mf_adam_continuation.json'),
                mf_adam_scale_selection='maximum utility in Exp2 momentum_scale history',
                legacy_exp3_selected=inputs['legacy_selected'],
                active_geometry_required=True, final_seeds=list(FINAL_SEEDS))


def save_rows(rows):
    fields = ('method', 'source_method', 'source_csv', 'stage', 'origin', 'seed', *PARAMETERS,
              'lr', 'eps_scale', 'num_bands', 'final_test_top1', 'result_dir')
    with (BASE / 'results/tuning_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    inputs, plan = read_inputs(), protocol()
    (BASE / 'results').mkdir(parents=True, exist_ok=True)
    write_json(BASE / 'results/input_audit.json', dict(input_files=inputs['input_files'],
        completed_exp3_trials=sum(r['origin'] == 'exp3' for r in inputs['rows']),
        imported_choices=inputs['choices'], exp2_utility_configs=inputs['adam'],
        exp2_utility_choices=inputs['adam_choices'],
        mf_adam_history=[r for r in inputs['rows'] if r['method'] == 'momentum_standard'],
        mf_adam_selection='maximum final_test_top1; Exp3b continuation pending; Exp2 C=1 was a fixed anchor',
        legacy_exp2_selected=inputs['legacy_exp2_selected'], legacy_selected=inputs['legacy_selected']))
    selected_path = BASE / 'results/selected_configs.json'
    rows = collect_rows(inputs)
    if selected_path.exists():
        previous = json.loads(selected_path.read_text())
        assert previous['status'] in ('pending_refinement', 'frozen')
        if previous['status'] == 'frozen':
            assert previous['mf_adam_tuning']['fixed_anchor_used'] is False
    if not selected_path.exists() or previous['status'] == 'pending_refinement':
        write_json(selected_path, selection(inputs, rows, frozen=False))
    save_rows(rows)
    final_path = BASE / 'results/final_summary.json'
    if not final_path.exists():
        write_json(final_path, dict(status='pending_full_experiment', final_seeds=list(FINAL_SEEDS),
                                   cells={}, note='No full training was started during implementation.'))
    from exp3b.replay import momentum_theory, write_csv, FIELDS
    write_csv(BASE / 'results/cancellation/momentum_theory.csv', momentum_theory(), FIELDS)
    standard = config(inputs['choices']['mf_muon_standard'])
    assert standard['muon_lr'] == .006 and standard['adam_lr'] == .012 and standard['max_grad_norm'] == 30.
    for lr in plan['standard_initial_lrs']:
        spec = RunSpec(method='mf_muon_standard', result_dir=f'exp3b/results/tuning/standard_{lr:g}',
                       **dict(standard, muon_lr=lr))
        write_json(BASE / f'specs/standard_{lr:g}.json', spec.mapping())
    for lr in plan['adam_standard']['initial_lrs']:
        values = dict(inputs['adam']['momentum_standard'], lr=lr)
        spec = RunSpec(method='momentum_standard', result_dir=f'exp3b/results/tuning/adam_standard_lr_{lr:g}_clip_{values["max_grad_norm"]:g}',
                       **{k: v for k, v in values.items() if k != 'num_bands'})
        write_json(BASE / f'specs/adam_standard_{lr:g}.json', spec.mapping())
    return inputs, plan


def collect_rows(inputs):
    rows = list(inputs['rows'])
    for file in sorted((BASE / 'results/tuning').glob('*/summary.json')):
        summary = json.loads(file.read_text())
        assert summary['status'] == 'completed' and not summary['smoke'] and summary['optimizer_steps'] == 250
        spec = RunSpec(**json.loads((file.parent / 'config.json').read_text())['spec'])
        from exp3b.spec import read_completed
        read_completed(spec)
        values = (dict(lr=spec.lr, max_grad_norm=spec.max_grad_norm, num_bands=4,
                       **(dict(eps_scale=spec.eps_scale) if spec.method == 'momentum_scale' else {}))
                  if spec.method in ADAM else {p: getattr(spec, p) for p in PARAMETERS})
        rows.append(dict(method=spec.method, source_method=spec.method, source_csv=str(file.relative_to(ROOT)),
                         stage='exp3b_refinement', origin='exp3b', seed=spec.seed,
                         result_dir=spec.result_dir, final_test_top1=summary['final_test_top1'],
                         **values))
    return rows


def evaluate_candidate(inputs, rows, method, values, label, gpus):
    existing = [r for r in rows if key(r['method'], r) == key(method, values)]
    if existing:
        return max(existing, key=lambda r: r['final_test_top1'])
    if method in ADAM:
        assert values['num_bands'] == 4
    spec = RunSpec(method=method, result_dir=f'exp3b/results/tuning/{label}',
                   **{k: v for k, v in values.items() if k != 'num_bands'})
    result = run_trials([spec], gpus, label)[0]
    row = dict(method=method, source_method=method,
               source_csv=str((spec.directory / 'summary.json').relative_to(ROOT)),
               stage='exp3b_refinement', origin='exp3b', seed=spec.seed,
               result_dir=spec.result_dir, final_test_top1=result['final_test_top1'], **values)
    rows.append(row)
    save_rows(rows)
    return row


def adam_lr_bracket(rows, clip, drop):
    candidates = [r for r in rows if r['method'] == 'momentum_standard' and r['max_grad_norm'] == clip]
    assert candidates
    best = max(candidates, key=lambda row: row['final_test_top1'])
    lower = [r for r in candidates if r['lr'] < best['lr'] and r['final_test_top1'] < best['final_test_top1'] - drop]
    upper = [r for r in candidates if r['lr'] > best['lr'] and r['final_test_top1'] < best['final_test_top1'] - drop]
    return best, lower, upper


def adam_standard_continuation(inputs, rows, plan):
    """Small two-sided LR bracket, then at most two clipping comparisons if LR moves."""
    policy = plan['adam_standard']
    initial = inputs['adam']['momentum_standard']
    clip, drop = initial['max_grad_norm'], policy['clear_drop_absolute_top1']
    observations = []
    def evaluate(lr, c):
        values = dict(lr=lr, max_grad_norm=c, num_bands=4)
        reused = any(key(r['method'], r) == key('momentum_standard', values) for r in rows)
        row = evaluate_candidate(inputs, rows, 'momentum_standard', values,
            f'adam_standard_lr_{lr:g}_clip_{c:g}', plan['gpus'])
        observations.append(dict(lr=lr, max_grad_norm=c, final_test_top1=row['final_test_top1'],
                                 reused=reused, result_dir=row['result_dir']))
        return row
    for lr in policy['initial_lrs']:
        _, lower, upper = adam_lr_bracket(rows, clip, drop)
        if lower and upper:
            break
        evaluate(lr, clip)
    for _ in range(policy['max_lr_extensions']):
        _, lower, upper = adam_lr_bracket(rows, clip, drop)
        if lower and upper:
            break
        rates = [r['lr'] for r in rows if r['method'] == 'momentum_standard' and r['max_grad_norm'] == clip]
        lr = min(rates) / policy['lr_extension_factor'] if not lower else max(rates) * policy['lr_extension_factor']
        evaluate(round(lr, 8), clip)
    best, lower, upper = adam_lr_bracket(rows, clip, drop)
    assert lower and upper, 'MF-Adam LR lacks a two-sided utility bracket within the small declared budget'
    moved = abs(best['lr'] / initial['lr'] - 1.) >= policy['lr_move_relative_threshold']
    if moved:
        for c in policy['clips']:
            evaluate(best['lr'], c)
    chosen = choose(rows, 'momentum_standard')
    summary = dict(status='completed',
        objective='maximum final_test_top1', initial_config=initial,
        initial_observation=inputs['adam_choices']['momentum_standard'], fixed_clip_lr_best=utility_config(best),
        lower_bracket=[dict(lr=r['lr'], final_test_top1=r['final_test_top1']) for r in lower],
        upper_bracket=[dict(lr=r['lr'], final_test_top1=r['final_test_top1']) for r in upper],
        lr_moved=moved, observations=observations, selected_config=utility_config(chosen),
        selected_observation=chosen, matched_label_priority=False, fixed_anchor_used=False)
    write_json(BASE / 'results/mf_adam_continuation.json', summary)
    return summary


def standard_continuation(inputs, rows, plan):
    base = config(inputs['choices']['mf_muon_standard'])
    threshold = plan['clear_drop_absolute_top1']
    best = inputs['choices']['mf_muon_standard']['final_test_top1']
    rates = list(plan['standard_initial_lrs'])
    observations = []
    bracket = None
    index = 0
    while index < len(rates):
        lr = rates[index]
        row = evaluate_candidate(inputs, rows, 'mf_muon_standard', dict(base, muon_lr=lr),
                                 f'standard_{lr:g}', plan['gpus'])
        observations.append(dict(lr=lr, final_test_top1=row['final_test_top1'], previous_best=best))
        if row['final_test_top1'] < best - threshold:
            bracket = lr
            break
        best = max(best, row['final_test_top1'])
        index += 1
        if index == len(rates):
            assert len(rates) < 2 + plan['max_standard_extensions'], (
                'No upper bracket within the declared small sequential budget; inspect utility observations')
            rates.append(round(rates[-1] * plan['standard_extension_factor'], 6))
    assert bracket is not None
    write_json(BASE / 'results/standard_bracket.json', dict(
        observations=observations, upper_bracket_lr=bracket, clear_drop_absolute_top1=threshold,
        chosen=config(choose(rows, 'mf_muon_standard')), selection='maximum final_test_top1'))


def refinement_candidates(active_config, standard_config, budget):
    """Three Muon LR points plus one Adam LR point near the final Standard basin."""
    center = standard_config['muon_lr']
    adam = standard_config['adam_lr']
    points = [(center, adam), (.75 * center, adam), (1.25 * center, adam), (center, .75 * adam)]
    return [dict(active_config, muon_lr=round(muon, 8), adam_lr=round(alr, 8))
            for muon, alr in points[:budget]]


def tune(inputs, plan):
    rows = collect_rows(inputs)
    adam_tuning = adam_standard_continuation(inputs, rows, plan)
    audit = json.loads((BASE / 'results/input_audit.json').read_text())
    audit.update(mf_adam_selection='maximum final_test_top1; Exp3b utility continuation completed; fixed anchor unused',
                 exp3b_mf_adam_tuning=adam_tuning)
    write_json(BASE / 'results/input_audit.json', audit)
    standard_continuation(inputs, rows, plan)
    standard = config(choose(rows, 'mf_muon_standard'))
    for method in ('mf_muon_normscale', 'mf_muon_spectralscale'):
        active_config = config(inputs['choices'][method])
        candidates = refinement_candidates(active_config, standard, plan['scale_refinement_budget'])
        for index, candidate in enumerate(candidates):
            evaluate_candidate(inputs, rows, method, candidate, f'{method}_refine_{index}', plan['gpus'])
    frozen = selection(inputs, rows, frozen=True)
    write_json(BASE / 'results/selected_configs.json', frozen)
    audit.update(exp3b_frozen_configs=frozen['configs'], exp3b_utility_provenance=frozen['provenance'])
    write_json(BASE / 'results/input_audit.json', audit)
    save_rows(rows)
    return frozen


def final_trials(frozen, plan):
    assert frozen['status'] == 'frozen'
    specs = [RunSpec(method=method, result_dir=f'exp3b/results/final/{method}/seed_{seed}', seed=seed,
                     capture=method == 'mf_muon_standard' and seed == FINAL_SEEDS[0],
                     **frozen['configs'][method]) for method in HYBRID for seed in FINAL_SEEDS]
    summaries = run_trials(specs, plan['gpus'], 'final')
    import statistics
    cells = {}
    for method in HYBRID:
        scores = [r['final_test_top1'] for r in summaries if r['method'] == method]
        cells[method] = dict(chosen_hyperparameters=frozen['configs'][method], seeds=list(FINAL_SEEDS),
                             final_test_top1=scores, mean=statistics.mean(scores), std=statistics.stdev(scores))
    write_json(BASE / 'results/final_summary.json', dict(status='completed', objective='final_test_top1',
                                                       cells=cells, selected_configs=frozen))
    return specs


def cancellation(frozen, plan, final_specs):
    from exp3b.replay import REPLAY_SEEDS
    assert plan['replay_seeds'] == list(REPLAY_SEEDS)
    sources = [RunSpec(method=method, result_dir=f'exp3b/results/source/{method}', seed=FINAL_SEEDS[0],
                      capture=True, **{k: v for k, v in frozen['configs'][method].items() if k != 'num_bands'})
               for method in ('momentum_standard', 'momentum_scale')]
    run_trials(sources, plan['gpus'], 'adam_sources')
    muon = next(spec for spec in final_specs if spec.capture)
    sources.append(muon)
    commands = [[sys.executable, '-m', 'exp3b.replay', '--source', str(spec.directory / 'signals'),
                 '--result', f'exp3b/results/cancellation/{name}']
                for spec, name in zip(sources, ('adam', 'adam_scale', 'muon'))]
    run_commands(commands, plan['gpus'], 'replay')
    run_commands([[sys.executable, '-m', 'exp3b.report']], plan['gpus'], 'report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Read inputs and write plans only; no training')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--cache-root', default='cache')
    args = parser.parse_args()
    assert args.gpus == '0,1,2,3'
    assert (ROOT / args.data_root).resolve() == ROOT / 'data'
    assert (ROOT / args.cache_root).resolve() == ROOT / 'cache'
    inputs, plan = prepare()
    if args.prepare:
        print('Prepared exp3b specs and imported utility configs; full experiment not started.')
        return
    frozen = tune(inputs, plan)
    finals = final_trials(frozen, plan)
    cancellation(frozen, plan, finals)


if __name__ == '__main__':
    main()
