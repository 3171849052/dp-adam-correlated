import pytest
from exp3b.experiment import refinement_candidates, selection, protocol, evaluate_candidate
from exp3b.history import choose, read_inputs
from exp3b.spec import PARAMETERS, RunSpec


def test_actual_history_active_geometries_and_exp2_utility_configs():
    inputs = read_inputs()
    assert inputs['choices']['mf_muon_standard']['muon_lr'] == .006
    assert inputs['choices']['mf_muon_normscale']['lambda_parallel'] == .1
    assert inputs['choices']['mf_muon_spectralscale']['kappa'] == 1.1
    assert inputs['adam']['momentum_standard'] == dict(lr=.005, max_grad_norm=1., num_bands=4)
    assert inputs['adam']['momentum_scale'] == dict(lr=.005, max_grad_norm=100., eps_scale=.1, num_bands=4)
    chosen = selection(inputs, inputs['rows'], False)
    assert chosen['status'] == 'pending_refinement'
    assert chosen['configs']['mf_muon_normscale']['lambda_parallel'] != 1


def test_utility_selection_ignores_cancellation():
    rows = [dict(method='mf_muon_standard', final_test_top1=.7, cancellation=.01),
            dict(method='mf_muon_standard', final_test_top1=.8, cancellation=100.)]
    assert choose(rows, 'mf_muon_standard') is rows[1]


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
