"""Sampled linearized noise multipliers; these are not Adam Jacobians/variances."""
import csv
import json
import statistics
from pathlib import Path
import numpy as np
import torch


def coordinate_indices(parameter_count):
    return np.sort(np.random.default_rng(0).choice(parameter_count, 2048, replace=False))


def mechanism_metrics(r, previous=None):
    r = np.asarray(r, dtype=np.float64)
    assert np.all(r > 0) and np.all(np.isfinite(r))
    quantiles = np.quantile(r, [.1, .5, .9, .99])
    mean, std = float(r.mean()), float(r.std())
    return dict(zip(('r_p10', 'r_p50', 'r_p90', 'r_p99'), map(float, quantiles)),
                r_mean=mean, r_std=std, r_cv=std / mean,
                r_anisotropy=float(quantiles[2] / quantiles[0]),
                temporal_drift=None if previous is None else
                float(np.median(np.abs(np.log(r) - np.log(previous)))))


class MechanismTrace:
    def __init__(self, parameters, scaled):
        parameters = list(parameters)
        self.indices = coordinate_indices(sum(p.numel() for p in parameters))
        self.slices = []
        offset = 0
        for p in parameters:
            indices = self.indices[(self.indices >= offset) & (self.indices < offset + p.numel())]
            if len(indices):
                self.slices.append((p, torch.as_tensor(indices - offset, device=p.device)))
            offset += p.numel()
        self.scaled = scaled
        self.r_trace, self.s_trace, self.metrics = [], [], []

    @torch.no_grad()
    def record(self, optimizer):
        """Called after Adam.step; _logical_scale is still the scale used this step."""
        beta2 = optimizer.param_groups[0]['betas'][1]
        eps = optimizer.param_groups[0]['eps']
        r_values, s_values = [], []
        for p, indices in self.slices:
            state = optimizer.state[p]
            step = int(state['step'].item())
            vhat = state['exp_avg_sq'].flatten()[indices] / (1 - beta2 ** step)
            preconditioner = 1 / (vhat.sqrt() + eps)
            scale = p._logical_scale.flatten()[indices] if self.scaled else torch.ones_like(preconditioner)
            r_values.append((preconditioner / scale).cpu().numpy())
            s_values.append(scale.cpu().numpy())
        r, s = np.concatenate(r_values), np.concatenate(s_values)
        previous = self.r_trace[-1] if self.r_trace else None
        self.metrics.append(dict(step=len(self.r_trace) + 1, **mechanism_metrics(r, previous)))
        self.r_trace.append(r)
        self.s_trace.append(s)

    def save(self, directory):
        np.savez(directory / 'mechanism_trace.npz', coordinate_indices=self.indices,
                 r_trace=np.stack(self.r_trace), s_trace=np.stack(self.s_trace))
        write_csv(directory / 'mechanism_metrics.csv', self.metrics)


def mf_distortion(r_trace, M, W):
    """Evaluate ||W diag(r) M||_F² using the actual trial matrices."""
    r = np.asarray(r_trace, dtype=np.float64)
    M, W = np.asarray(M, dtype=np.float64), np.asarray(W, dtype=np.float64)
    assert M.shape == W.shape == (r.shape[0], r.shape[0])
    H = (W.T @ W) * (M @ M.T)
    actual_sq = np.sum(r * (H @ r), axis=0)
    ideal_sq = np.mean(r ** 2, axis=0) * H.sum()
    assert np.all(actual_sq > 0) and np.all(ideal_sq > 0)
    ratio = np.sqrt(actual_sq / ideal_sq)
    logabs = np.abs(np.log(ratio))
    return dict(ratio_p10=float(np.quantile(ratio, .1)),
                ratio_p50=float(np.median(ratio)), ratio_p90=float(np.quantile(ratio, .9)),
                logabs_mean=float(logabs.mean()), logabs_median=float(np.median(logabs)))


def factorial_effects(accuracy):
    gains = {f'G_{noise}': accuracy[f'{noise}_scale'] - accuracy[f'{noise}_standard']
             for noise in ('iid', 'prefix', 'momentum')}
    return dict(gains, I_prefix=gains['G_prefix'] - gains['G_iid'],
                I_momentum=gains['G_momentum'] - gains['G_iid'])


def mean_std(values):
    return dict(mean=statistics.mean(values), sample_std=statistics.stdev(values))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
