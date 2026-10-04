"""Read historical utility choices, including active (non-identity) geometry."""
import csv
import json
import math
from exp3b import ROOT
from exp3b.spec import HYBRID, PARAMETERS


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
    return method, tuple(values[p] for p in PARAMETERS)


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
    # Read both Exp2 selected configs and final summary. They must agree; matched
    # Standard results are deliberately excluded from the principal comparison.
    exp2_selected = json.loads((ROOT / 'exp2/results/selected_configs.json').read_text())
    exp2_final = json.loads((ROOT / 'exp2/results/final_summary.json').read_text())['cells']
    adam = {}
    for method in ('momentum_standard', 'momentum_scale'):
        config = exp2_final[method]['chosen_hyperparameters']
        assert config == exp2_selected[method]
        assert config['num_bands'] == 4
        adam[method] = config
    search = list(csv.DictReader((ROOT / 'exp2/results/search_summary.csv').open()))
    scale_rows = [r for r in search if r['method'] == 'momentum_scale']
    best_scale = max(scale_rows, key=lambda r: float(r['final_test_top1']))
    assert float(best_scale['lr']) == adam['momentum_scale']['lr']
    assert float(best_scale['max_grad_norm']) == adam['momentum_scale']['max_grad_norm']
    return dict(rows=rows, choices=choices, adam=adam, legacy_selected=legacy,
                input_files=['exp3/results/search_history.json', 'exp3/results/selected_configs.json',
                             'exp2/results/selected_configs.json', 'exp2/results/final_summary.json',
                             'exp2/results/search_summary.csv'])
