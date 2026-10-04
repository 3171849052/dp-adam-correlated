"""Validate every cumulative step, combine methods, and export publication figures."""
import argparse
import csv
import json

import numpy as np
from exp3b import BASE, offline_runtime
from exp3b.replay import FIELDS, momentum_theory, write_csv
from exp3b.spec import output_path, write_json

offline_runtime()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def validate(rows, method, steps):
    assert len(rows) == steps
    assert [int(row['step']) for row in rows] == list(range(1, steps + 1))
    assert all(row['method'] == method for row in rows)
    for row in rows:
        assert set(row) == set(FIELDS)
        assert all(np.isfinite(float(row[key])) for key in FIELDS if key != 'method')
        assert float(row['rmse_iid_mean']) > 0
    return rows


def plot(directory, groups):
    colors = ('#4c566a', '#0072b2', '#e69f00', '#009e73')
    with plt.rc_context({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False}):
        for kind in ('cumulative_rmse', 'cancellation_curve', 'relative_to_momentum'):
            fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
            for (method, rows), color in zip(groups.items(), colors):
                steps = np.asarray([int(row['step']) for row in rows])
                if kind == 'cumulative_rmse':
                    for noise, style in (('mf', '-'), ('iid', '--')):
                        mean = np.asarray([float(row[f'rmse_{noise}_mean']) for row in rows])
                        std = np.asarray([float(row[f'rmse_{noise}_std']) for row in rows])
                        ax.plot(steps, mean, style, color=color, label=f'{method} / {noise.upper()}')
                        ax.fill_between(steps, np.maximum(0., mean - std), mean + std, color=color, alpha=.1)
                    ax.set_ylabel('Cumulative pre-LR update RMSE')
                else:
                    field = 'cancellation_mean' if kind == 'cancellation_curve' else 'relative_to_momentum'
                    mean = np.asarray([float(row[field]) for row in rows])
                    std = np.asarray([float(row['cancellation_std']) for row in rows])
                    if kind == 'relative_to_momentum':
                        baseline = np.asarray([float(row['cancellation_mean']) for row in groups['MF-Momentum']])
                        std = std / baseline
                    ax.plot(steps, mean, color=color, label=method)
                    ax.fill_between(steps, np.maximum(0., mean - std), mean + std, color=color, alpha=.15)
                    ax.set_ylabel('MF / IID cumulative RMSE' if kind == 'cancellation_curve' else 'Cancellation / Momentum cancellation')
            if kind == 'cumulative_rmse':
                ax.text(.01, .99, 'Momentum theory: unit innovations; nonlinear: source-calibrated innovations',
                        transform=ax.transAxes, va='top', fontsize=7)
            else:
                ax.axhline(1., color='gray', linestyle=':', linewidth=.8)
            ax.set_xlabel('Logical step')
            ax.set_xlim(1, len(next(iter(groups.values()))))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, ncol=2 if kind == 'cumulative_rmse' else 1)
            for extension in ('png', 'pdf'):
                fig.savefig(directory / f'{kind}.{extension}', dpi=200)
            plt.close(fig)


def build(directory=BASE / 'results/cancellation', steps=250):
    directory = output_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    theory = validate(momentum_theory(steps), 'MF-Momentum', steps)
    write_csv(directory / 'momentum_theory.csv', theory, FIELDS)
    groups = {'MF-Momentum': theory}
    metadata = {}
    for name, method in (('adam', 'MF-Adam'), ('adam_scale', 'MF-Adam-Scale'), ('muon', 'MF-Muon')):
        with (directory / f'{name}.csv').open() as stream:
            rows = [{key: value if key == 'method' else int(value) if key == 'step' else float(value)
                     for key, value in row.items()} for row in csv.DictReader(stream)]
        validate(rows, method, steps)
        for index, row in enumerate(rows):
            expected = float(row['cancellation_mean']) / theory[index]['cancellation_mean']
            assert np.isclose(float(row['relative_to_momentum']), expected, rtol=1e-12)
        groups[method] = rows
        metadata[method] = json.loads((directory / f'{name}.json').read_text())
        assert metadata[method]['steps'] == steps and metadata[method]['support_count'] == 48
    supports = [meta['support'] for meta in metadata.values()]
    assert all(value == supports[0] for value in supports)
    assert len({meta['dimension'] for meta in metadata.values()}) == 1
    assert len({tuple(meta['replay_seeds']) for meta in metadata.values()}) == 1
    combined = [row for rows in groups.values() for row in rows]
    write_csv(directory / 'combined.csv', combined, FIELDS)
    write_json(directory / 'summary.json', dict(status='completed', steps=steps,
        methods=list(groups), rows=len(combined), nonlinear_support=supports[0],
        nonlinear_dimension=metadata['MF-Muon']['dimension'],
        metric='delta_u = complete noisy pre-LR update - complete clean pre-LR update; E_t=sum(delta_u); RMSE=sqrt(sum(E_t^2)/d)',
        cancellation='per paired seed C_t=RMSE_MF/RMSE_IID; reported mean/std across replay seeds',
        relative_to_momentum='mean(C_t_optimizer) / exact(C_t_momentum)',
        momentum='exact row norms of W D and W, W=L H_0.9, T=250, num_bands=4; RMSE uses unit innovation scale',
        iid='cancellation control only; exactly the MF innovation scale, no IID privacy calibration',
        adam_scale='clean = q^S / S; common realized clipped signal; inverse S only on added noise',
        muon='Nesterov beta=.95, Frobenius normalization, NS5, shape factor; only 48 hidden matrices',
        secondary='parameter RMSE = pre-LR RMSE * fixed actual source LR; Momentum theoretical LR=1',
        sources=metadata, final={method: rows[-1] for method, rows in groups.items()},
        full_experiment=steps == 250 and all(not meta['source_manifest']['smoke'] for meta in metadata.values())))
    plot(directory, groups)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default=str(BASE / 'results/cancellation'))
    args = parser.parse_args()
    build(args.directory)


if __name__ == '__main__':
    main()
