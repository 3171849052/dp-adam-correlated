import pytest
from exp3b.experiment import (adam_standard_continuation, cancellation, refinement_candidates,
                             selection, protocol, evaluate_candidate)
from exp3b.history import choose, read_inputs, utility_config
from exp3b.spec import PARAMETERS, RunSpec


def test_actual_history_active_geometries_and_exp2_utility_configs():
    inputs = read_inputs()
    assert inputs['choices']['mf_muon_standard']['muon_lr'] == .006
    assert inputs['choices']['mf_muon_normscale']['lambda_parallel'] == .1
    assert inputs['choices']['mf_muon_spectralscale']['kappa'] == 1.1
    assert inputs['adam']['momentum_standard'] == dict(lr=.005, max_grad_norm=30., num_bands=4)
    assert inputs['adam']['momentum_scale'] == dict(lr=.005, max_grad_norm=100., eps_scale=.1, num_bands=4)
    chosen = selection(inputs, inputs['rows'], False)
    assert chosen['status'] == 'pending_refinement'
    assert chosen['configs']['mf_muon_normscale']['lambda_parallel'] != 1
    assert chosen['configs']['momentum_standard']['max_grad_norm'] == 30.
    assert chosen['mf_adam_tuning']['fixed_anchor_used'] is False
    best = inputs['adam_choices']['momentum_standard']
    assert best['final_test_top1'] == .7361
    assert best['source_method'] == 'momentum_standard_matched'
    assert best['source_csv'] == 'exp2/results/matched_search_summary.csv'
    assert inputs['legacy_exp2_selected']['momentum_standard']['max_grad_norm'] == 1.
    assert len([r for r in inputs['rows'] if r['method'] == 'momentum_standard']) >= 6


def test_utility_selection_ignores_cancellation():
    rows = [dict(method='mf_muon_standard', final_test_top1=.7, cancellation=.01),
            dict(method='mf_muon_standard', final_test_top1=.8, cancellation=100.)]
    assert choose(rows, 'mf_muon_standard') is rows[1]
    adam = [dict(method='momentum_standard', source_method='momentum_standard_matched',
                 final_test_top1=.7, max_grad_norm=30., cancellation=.01),
            dict(method='momentum_standard', source_method='momentum_standard',
                 final_test_top1=.8, max_grad_norm=10., cancellation=100.)]
    assert choose(adam, 'momentum_standard') is adam[1]


def test_refinements_are_bounded_near_standard_and_keep_geometry():
    active = dict(muon_lr=.006, adam_lr=.012, max_grad_norm=30., lambda_parallel=.1, kappa=4., rho=.1)
    standard = dict(active, muon_lr=.012, lambda_parallel=1.)
    candidates = refinement_candidates(active, standard, 4)
    assert len(candidates) == 4
    assert {c['muon_lr'] for c in candidates} == {.009, .012, .015}
    assert all(c['lambda_parallel'] == .1 and c['max_grad_norm'] == 30. for c in candidates)
    assert protocol()['gpus'] == [0, 1, 2, 3]


def test_exact_completed_configuration_is_reused_without_training():
    inputs = read_inputs()
    row = inputs['choices']['mf_muon_standard']
    returned = evaluate_candidate(inputs, inputs['rows'], row['method'],
        {p: row[p] for p in PARAMETERS}, 'must_not_launch', [0, 1, 2, 3])
    assert returned is row


def test_completed_adam_configuration_reuses_matched_observation(monkeypatch):
    import exp3b.experiment as experiment
    inputs = read_inputs()
    def must_not_launch(*args):
        pytest.fail('Completed same MF-Adam mechanism/configuration must be reused')
    monkeypatch.setattr(experiment, 'run_trials', must_not_launch)
    returned = evaluate_candidate(inputs, inputs['rows'], 'momentum_standard',
        dict(lr=.005, max_grad_norm=30., num_bands=4), 'must_not_launch', [0, 1, 2, 3])
    assert returned is inputs['adam_choices']['momentum_standard']
    assert returned['final_test_top1'] == .7361


@pytest.mark.parametrize('moved', [False, True])
def test_adam_small_bracket_clipping_and_utility_selection(monkeypatch, moved):
    import exp3b.experiment as experiment
    inputs = read_inputs()
    rows = list(inputs['rows'])
    scores = ({(.003, 30.): .75, (.007, 30.): .73, (.002, 30.): .72,
               (.003, 10.): .76, (.003, 100.): .74} if moved else
              {(.003, 30.): .72, (.007, 30.): .724})
    calls, written = [], {}
    def trial(specs, gpus, label):
        spec = specs[0]
        calls.append((spec.lr, spec.max_grad_norm))
        return [dict(final_test_top1=scores[(spec.lr, spec.max_grad_norm)])]
    monkeypatch.setattr(experiment, 'run_trials', trial)
    monkeypatch.setattr(experiment, 'save_rows', lambda rows: None)
    monkeypatch.setattr(experiment, 'write_json', lambda path, value: written.update(value))
    adam_standard_continuation(inputs, rows, protocol())
    winner = choose(rows, 'momentum_standard')
    if moved:
        assert calls == [(.003, 30.), (.007, 30.), (.002, 30.), (.003, 10.), (.003, 100.)]
        assert utility_config(winner) == dict(lr=.003, max_grad_norm=10., num_bands=4)
    else:
        assert calls == [(.003, 30.), (.007, 30.)]
        assert winner['lr'] == .005 and winner['max_grad_norm'] == 30.
    assert written['status'] == 'completed' and written['lr_moved'] == moved
    assert written['matched_label_priority'] is False
    assert written['selected_config'] == utility_config(winner)


def test_adam_existing_bracket_stops_without_training(monkeypatch):
    import exp3b.experiment as experiment
    inputs = read_inputs()
    base = inputs['adam_choices']['momentum_standard']
    rows = list(inputs['rows']) + [dict(base, lr=.003, final_test_top1=.7), dict(base, lr=.007, final_test_top1=.71)]
    monkeypatch.setattr(experiment, 'run_trials', lambda *args: pytest.fail('Bracket already exists'))
    written = {}
    monkeypatch.setattr(experiment, 'write_json', lambda path, value: written.update(value))
    adam_standard_continuation(inputs, rows, protocol())
    assert written['observations'] == [] and written['selected_config']['max_grad_norm'] == 30.


def test_mf_adam_source_uses_exp3b_utility_continuation_winner(monkeypatch):
    import exp3b.experiment as experiment
    inputs = read_inputs()
    winner = dict(inputs['adam_choices']['momentum_standard'], lr=.003, max_grad_norm=10.,
                  final_test_top1=.8, origin='exp3b', source_method='momentum_standard')
    frozen = selection(inputs, inputs['rows'] + [winner], True)
    observed = []
    monkeypatch.setattr(experiment, 'run_trials', lambda specs, *args: observed.extend(specs))
    monkeypatch.setattr(experiment, 'run_commands', lambda *args: None)
    muon = RunSpec(method='mf_muon_standard', result_dir='exp3b/results/test_muon_source',
                   seed=20261011, capture=True, **frozen['configs']['mf_muon_standard'])
    cancellation(frozen, protocol(), [muon])
    adam = next(spec for spec in observed if spec.method == 'momentum_standard')
    assert adam.capture and adam.lr == .003 and adam.max_grad_norm == 10.
    assert frozen['provenance']['momentum_standard']['origin'] == 'exp3b'


def test_adam_scale_selection_keeps_maximum_utility(monkeypatch):
    inputs = read_inputs()
    winner = dict(inputs['adam_choices']['momentum_scale'], lr=.007, max_grad_norm=200., final_test_top1=.8)
    selected = selection(inputs, inputs['rows'] + [winner], False)
    assert selected['configs']['momentum_scale'] == dict(lr=.007, max_grad_norm=200., num_bands=4, eps_scale=.1)


def test_invalid_specs_fail_without_fallback():
    with pytest.raises(AssertionError):
        RunSpec(method='momentum_scale', result_dir='exp2/results/forbidden')
    with pytest.raises(AssertionError):
        RunSpec(method='mf_muon_normscale', result_dir='exp3b/results/x', capture=True)
    with pytest.raises(AssertionError):
        RunSpec(method='momentum_standard', result_dir='exp3b/results/x', data_root='other')


@pytest.mark.parametrize('scores,expected', [([.795, .75], [.009, .012]),
                                          ([.70], [.009]),
                                          ([.8, .81, .75], [.009, .012, .016])])
def test_sequential_standard_stops_at_first_clear_drop(monkeypatch, scores, expected):
    import exp3b.experiment as experiment
    inputs = read_inputs()
    rows = list(inputs['rows'])
    observations = []
    def evaluate(inputs, rows, method, values, label, gpus):
        observations.append(values['muon_lr'])
        row = dict(method=method, final_test_top1=scores[len(observations) - 1], **values)
        rows.append(row)
        return row
    monkeypatch.setattr(experiment, 'evaluate_candidate', evaluate)
    monkeypatch.setattr(experiment, 'write_json', lambda path, values: None)
    experiment.standard_continuation(inputs, rows, protocol())
    assert observations == expected
