"""Read historical utility choices, including active (non-identity) geometry."""
import csv
import json
import math
from exp3b import ROOT
from exp3b.spec import ADAM, HYBRID, PARAMETERS


def active(row):
    method = row['method']
    return (row['lambda_parallel'] != 1. if method == 'mf_muon_normscale' else
            row['kappa'] != 1. if method == 'mf_muon_spectralscale' else True)


def choose(rows, method, require_active=False):
    candidates = [r for r in rows if r['method'] == method and
                  (not require_active or active(r))]
    assert candidates
    assert all(math.isfinite(r['final_test_top1']) for r in candidates)
    return max(candidates, key=lambda r: r['final_test_top1'])


def key(method, values):
    parameters = ('lr', 'max_grad_norm') if method in ADAM else PARAMETERS
    if method == 'momentum_scale':
        parameters += ('eps_scale',)
    return method, tuple(values[p] for p in parameters)


def utility_config(row):
    if row['method'] in ADAM:
        values = {p: row[p] for p in ('lr', 'max_grad_norm', 'num_bands')}
        if row['method'] == 'momentum_scale':
            values['eps_scale'] = row['eps_scale']
        return values
    return {p: row[p] for p in PARAMETERS}


def read_adam_observations():
    """Canonicalize matched Standard as the same mechanism; labels never rank trials."""
    rows = []
    for csv_name in ('search_summary.csv', 'matched_search_summary.csv'):
        csv_path = ROOT / 'exp2/results' / csv_name
        with csv_path.open() as stream:
            records = list(csv.DictReader(stream))
        for record in records:
            if record['method'] not in ('momentum_standard', 'momentum_standard_matched', 'momentum_scale'):
                continue
            directory = (ROOT / record['result_dir']).resolve()
            assert directory.is_relative_to(ROOT / 'exp2/results')
            summary = json.loads((directory / 'summary.json').read_text())
            assert summary['status'] == 'completed' and not summary['smoke']
            assert summary['optimizer_steps'] == 250 and len(summary['epochs']) == 5
            assert summary['method'] == record['method']
            assert summary['seed'] == int(record['seed']) == 20261001
            assert summary['noise'] == record['noise'] == 'momentum_bandinvmf'
            assert summary['num_bands'] == int(record['num_bands']) == 4
            assert summary['lr'] == float(record['lr'])
            assert summary['max_grad_norm'] == float(record['max_grad_norm'])
            assert summary['final']['test_top1'] == float(record['final_test_top1'])
            method = record['method'].removesuffix('_matched')
            row = dict(method=method, source_method=record['method'], stage='exp2_utility_history',
                       source_csv=str(csv_path.relative_to(ROOT)), origin='exp2', seed=20261001,
                       result_dir=str(directory.relative_to(ROOT)),
                       lr=summary['lr'], max_grad_norm=summary['max_grad_norm'], num_bands=4,
                       final_test_top1=summary['final']['test_top1'])
            if method == 'momentum_scale':
                assert summary['eps_scale'] == float(record['eps_scale']) == .1
                row['eps_scale'] = .1
            assert math.isfinite(row['final_test_top1'])
            rows.append(row)
    assert any(row['method'] == 'momentum_standard' for row in rows)
    assert any(row['method'] == 'momentum_scale' for row in rows)
    return rows


def read_inputs():
    history_path = ROOT / 'exp3/results/search_history.json'
    history = json.loads(history_path.read_text())
    legacy = json.loads((ROOT / 'exp3/results/selected_configs.json').read_text())
    assert history['protocol']['tuning_seed'] == 20261001
    assert set(legacy) == set(HYBRID)
    rows = []
    for record in history['trials']:
        if record['status'] != 'completed' or record['reused_identity']:
            continue
        assert record['seed'] == 20261001
        directory = (ROOT / record['result_dir']).resolve()
        assert directory.is_relative_to(ROOT / 'exp3/results')
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['status'] == 'completed' and not summary['smoke']
        assert summary['optimizer_steps'] == 250 and len(summary['epoch_test_top1']) == 5
        assert summary['final_test_top1'] == record['final_test_top1']
        assert all(summary[p] == record[p] for p in PARAMETERS)
        rows.append(dict(method=record['method'], stage='exp3_history', origin='exp3',
                         seed=20261001, result_dir=str(directory.relative_to(ROOT)),
                         final_test_top1=record['final_test_top1'],
                         **{p: record[p] for p in PARAMETERS}))
    choices = {m: choose(rows, m, m.endswith(('normscale', 'spectralscale'))) for m in HYBRID}
    # Exp2 Standard was a fixed anchor. Import actual utility observations,
    # including matched-search trials, and rank all by utility alone.
    exp2_selected = json.loads((ROOT / 'exp2/results/selected_configs.json').read_text())
    exp2_final = json.loads((ROOT / 'exp2/results/final_summary.json').read_text())['cells']
    for method in ('momentum_standard', 'momentum_scale'):
        config = exp2_final[method]['chosen_hyperparameters']
        assert config == exp2_selected[method]
        assert config['num_bands'] == 4
    adam_rows = read_adam_observations()
    adam_choices = {m: choose(adam_rows, m) for m in ADAM}
    adam = {m: utility_config(row) for m, row in adam_choices.items()}
    return dict(rows=rows + adam_rows, choices=choices, adam=adam, adam_choices=adam_choices,
                legacy_exp2_selected=exp2_selected, legacy_selected=legacy,
                input_files=['exp3/results/search_history.json', 'exp3/results/selected_configs.json',
                             'exp2/results/selected_configs.json', 'exp2/results/final_summary.json',
                             'exp2/results/search_summary.csv', 'exp2/results/matched_search_summary.csv'])
