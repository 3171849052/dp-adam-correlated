"""Reports use only successful real CV trials, never synthetic measurements."""
import csv
import json
import numpy as np
from exp11 import RESULTS
from exp11.config import save_json


def completed():
    rows = [json.loads(path.read_text()) for path in (RESULTS / 'cv').glob('*/summary.json')]
    return sorted([r for r in rows if r['status'] == 'completed'], key=lambda r: r['lambda_value'])


def rank(rows):
    return sorted(rows, key=lambda r: (-r['accuracy'], r['validation_loss'], r['lambda_value']))


def generate(total=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = completed()
    if not rows:
        return
    summary = []
    for r in rows:
        m = r['matrix_optimization']
        summary.append(dict(lambda_value=r['lambda_value'], validation_accuracy=r['accuracy'],
                            validation_loss=r['validation_loss'], seed=r['seed'],
                            **{k: m[k] for k in ('normalized_J1', 'J1', 'J2_variance', 'J2_bias_sq', 'J2_mse',
                                                'normalized_J2_variance', 'normalized_J2_bias_sq', 'normalized_J2_mse',
                                                'normalized_objective', 'sensitivity', 'sigma_avg')}))
    with (RESULTS / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    positive = [r for r in rows if r['lambda_value'] > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r['lambda_value'] for r in positive], [100 * r['accuracy'] for r in positive], 'o-')
    baseline = next((r for r in rows if r['lambda_value'] == 0), None)
    if baseline:
        ax.axhline(100 * baseline['accuracy'], linestyle='--', color='gray', label='lambda=0 (MF J1 optimum)')
        ax.legend()
    ax.set(xscale='log', xlabel='lambda', ylabel='Internal validation accuracy (%)', title='CIFAR-100, seed 20261101')
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(RESULTS / 'lambda_validation_accuracy.png', dpi=180)
    fig.savefig(RESULTS / 'lambda_validation_accuracy.pdf')
    plt.close(fig)
    points = np.array([[r['matrix_optimization']['normalized_J1'], r['matrix_optimization']['normalized_J2_mse']] for r in rows])
    frontier = [i for i, p in enumerate(points)
                if not any(np.all(q <= p) and np.any(q < p) for q in points)]
    frontier.sort(key=lambda i: points[i, 0])
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=np.log10([max(r['lambda_value'], 1e-5) for r in rows]), cmap='viridis')
    ax.plot(points[frontier, 0], points[frontier, 1], color='gray', alpha=.6)
    best = rank(rows)[0]
    bm = best['matrix_optimization']
    ax.scatter([bm['normalized_J1']], [bm['normalized_J2_mse']], marker='*', s=180, color='red', label='Best validation lambda')
    ax.scatter([1.], [1.], marker='x', color='black', label='IID reference (not trained)')
    ax.set(xlabel='J1 / J1(IID)', ylabel='J2 MSE / J2 MSE(IID)', title='Numerical MF solutions and nondominated frontier')
    fig.colorbar(scatter, ax=ax, label='log10(lambda); lambda=0 displayed at -5')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(RESULTS / 'J1_J2_pareto.png', dpi=180)
    fig.savefig(RESULTS / 'J1_J2_pareto.pdf')
    plt.close(fig)
    status = 'complete' if total is not None and len(rows) == total else 'in progress'
    save_json(RESULTS / 'best.json', dict(status=status, completed_trainings=len(rows), planned_trainings=total,
                                       lambda_value=best['lambda_value'], validation_accuracy=best['accuracy'],
                                       validation_loss=best['validation_loss'], matrix_optimization=bm))
    lines = ['# Exp11: Adam-aware MF with second-moment MSE', '',
             f'Status: **{status}**. Completed real five-epoch CV trainings: {len(rows)} / {total if total is not None else "30 + up to 8"}.', '',
             f'Best observed lambda: **{best["lambda_value"]:.12g}**; final internal validation accuracy '
             f'**{100 * best["accuracy"]:.2f}%**; loss **{best["validation_loss"]:.8g}**.', '',
             '## Protocol', '',
             'CIFAR-100 only: 45,000 training / 5,000 internal validation, exactly the Exp10 split (seed 20261008). '
             'Offline cached pretrained ViT-Tiny, all parameters trained for five epochs / 225 logical steps. '
             'PyTorch Adam: lr=0.003, betas=(0.9,0.999), eps=1e-8, weight_decay=0; '
             'standard Euclidean per-example clipping C=24, logical batch=1000, physical batch=250. '
             'Physical GPUs 0,1,2,3, at most one CV worker per GPU (updated at user request). '
             'The already-running lambda=0 GPU-0 worker is preserved; gpu_transition.json records the exact source transition. '
             'All CV runs use seed 20261101 with identical initialization, '
             'ordering and augmentation RNG, independent innovation RNG seed=20261102. '
             'Only the CIFAR training pickle is loaded; the official test set is never read.', '',
             '## Objective and noise calibration', '',
             'D=I+d1 L+d2 L^2+d3 L^3. J1 uses the complete non-Toeplitz cumulative bias-corrected momentum workload '
             '(omitting the common 1-beta1 factor). J2_mse=J2_bias_sq+J2_variance uses the full bias-corrected '
             'Gaussian-square second moment at beta2=0.999, including truncated startup rows. '
             'The minimized objective is J1/J1(IID)+lambda J2_mse/J2_mse(IID). '
             'Both denominators use d=(1,0,0,0), never Momentum-MF. '
             'All J2 metrics and sigma_avg use C=24 and averaged-gradient noise; J1 follows Exp9 sensitivity-squared units. '
             'Nelder-Mead starts from Exp9 BISR, IID and available neighboring lambda solutions; '
             'these are numerical local optima, with no global-optimality claim.', '',
             'Each run uses Exp9 fixed-five-participation absolute-Gram sensitivity and GDP epsilon=8, delta=1e-5 '
             'under add/remove-zero-out adjacency, without sampling amplification. '
             'Internal validation is assumed public. The 38-run selection pipeline is not claimed to have total epsilon=8.', '',
             f'Best coefficients: `{bm["coefficients"]}`. Sensitivity={bm["sensitivity"]:.10g}, sigma_avg={bm["sigma_avg"]:.10g}.', '',
             f'Best noise metrics: J1={bm["J1"]:.10g}, J1/IID={bm["normalized_J1"]:.10g}; '
             f'J2_variance={bm["J2_variance"]:.10g}, J2_bias_sq={bm["J2_bias_sq"]:.10g}, '
             f'J2_mse={bm["J2_mse"]:.10g}, J2_mse/IID={bm["normalized_J2_mse"]:.10g}.', '',
             '## Search and limitations', '',
             'Round 1: lambda=0 plus 29 points 10^(k/4), k=-16,...,12 (1e-4 to 1e3). '
             'Round 2: up to eight distinct logarithmic refinements around the two best positive lambdas, '
             'ranked by final internal accuracy, then loss, then lambda. Refinement decisions are saved before training. '
             'No learning-rate/clipping search and no NLP runs. The lambda=0 training is the optimized pure-J1 MF baseline; '
             'IID is the mathematical normalization reference only.', '',
             'A single seed does not quantify seed-to-seed uncertainty or establish statistical significance. '
             'Selecting and reporting on the same internal validation set introduces selection optimism. '
             'This objective concerns only the pure Gaussian-noise square term, not all signal/noise cross terms '
             'or the complete nonlinear Adam update. No official-test generalization claim is made.', '',
             'Mathematical tests, source provenance and GPU smoke evidence are saved under results/. '
             'Each completed training has config.json, train.log, steps.csv, metrics.csv, train_order.npy, '
             'matrices.npz and summary.json. Incomplete runs are not included in the result table.', '',
             '## Complete observed results', '',
             '| lambda | final validation accuracy (%) | validation loss | J1/IID | J2 variance | J2 bias squared | J2 MSE | MSE/IID |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["lambda_value"]:.12g} | {100*r["validation_accuracy"]:.2f} | {r["validation_loss"]:.8g} '
                     f'| {r["normalized_J1"]:.8g} | {r["J2_variance"]:.8g} | {r["J2_bias_sq"]:.8g} '
                     f'| {r["J2_mse"]:.8g} | {r["normalized_J2_mse"]:.8g} |')
    (RESULTS / 'final_report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    generate()
