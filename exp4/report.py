"""Produce trial CSV and mean/sample-standard-deviation final summaries."""
from exp4.runtime import EXP, output_path
import argparse
import csv
import json
from pathlib import Path
import statistics

from exp4.config import METHODS


def trial_rows(trials):
    rows = []
    for trial in trials:
        summary = json.loads((Path(trial['result_dir']) / 'summary.json').read_text())
        assert summary['status'] == 'completed' and not summary['smoke']
        assert summary['optimizer_steps'] == summary['noise_steps'] == 250
        rows.append(dict(method=summary['method'], seed=summary['seed'], lr=summary['lr'],
                         update_clip_norm=summary['update_clip_norm'],
                         train_loss=summary['final']['train_loss'], test_loss=summary['final']['test_loss'],
                         test_top1=summary['final']['test_top1'],
                         epoch_clip_fraction=summary['final']['epoch_clip_fraction'],
                         gdp_epsilon=summary['final']['gdp_epsilon'], result_dir=trial['result_dir']))
    return rows


def write_csv(path, rows):
    with output_path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_search_report(trials):
    write_csv(EXP / 'results/search_summary.csv', trial_rows(trials))


def write_final_report(trials, root):
    root = output_path(root)
    rows = trial_rows(trials)
    write_csv(root / 'trials.csv', rows)
    write_csv(root / 'final_multiseed.csv', rows)
    aggregates = []
    for method in METHODS:
        group = [r for r in rows if r['method'] == method]
        assert sorted(r['seed'] for r in group) == [20261011, 20261012, 20261013]
        aggregate = dict(method=method, seeds=3)
        for key in ('test_top1', 'test_loss', 'train_loss', 'epoch_clip_fraction'):
            aggregate[key + '_mean'] = statistics.mean(r[key] for r in group)
            aggregate[key + '_std'] = statistics.stdev(r[key] for r in group)
        aggregates.append(aggregate)
    write_csv(root / 'aggregate.csv', aggregates)
    (root / 'final_summary.json').write_text(json.dumps(dict(
        total_trials=6, seeds=[20261011, 20261012, 20261013], std_ddof=1,
        methods={r['method']: dict(mean_test_top1=r['test_top1_mean'],
                                   sample_std_test_top1=r['test_top1_std'], **r) for r in aggregates}),
        indent=2, allow_nan=False))
    (root / 'report.md').write_text(
        '| method | test top1 mean ± sample std | test loss mean ± sample std |\n'
        '|---|---:|---:|\n' + ''.join(
            f"| {r['method']} | {r['test_top1_mean']:.4f} ± {r['test_top1_std']:.4f} | "
            f"{r['test_loss_mean']:.4f} ± {r['test_loss_std']:.4f} |\n" for r in aggregates))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=Path, required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    args = parser.parse_args()
    write_final_report(json.loads(args.trials.read_text()), args.result_root)
