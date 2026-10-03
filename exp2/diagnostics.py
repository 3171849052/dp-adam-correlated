"""Paper approximation and frozen-observed-v operators, not exact Adam Jacobians."""
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
        self.p_trace, self.r_trace, self.s_trace, self.metrics = [], [], [], []

    @torch.no_grad()
    def record(self, optimizer):
        """Called after Adam.step; _logical_scale is still the scale used this step."""
        beta2 = optimizer.param_groups[0]['betas'][1]
        eps = optimizer.param_groups[0]['eps']
        p_values, r_values, s_values = [], [], []
        for p, indices in self.slices:
            state = optimizer.state[p]
            step = int(state['step'].item())
            vhat = state['exp_avg_sq'].flatten()[indices] / (1 - beta2 ** step)
            preconditioner = 1 / (vhat.sqrt() + eps)
            scale = p._logical_scale.flatten()[indices] if self.scaled else torch.ones_like(preconditioner)
            r_values.append((preconditioner / scale).cpu().numpy())
            p_values.append(preconditioner.cpu().numpy())
            s_values.append(scale.cpu().numpy())
        r, s = np.concatenate(r_values), np.concatenate(s_values)
        previous = self.r_trace[-1] if self.r_trace else None
        self.metrics.append(dict(step=len(self.r_trace) + 1, **mechanism_metrics(r, previous)))
        self.r_trace.append(r)
        self.s_trace.append(s)
        self.p_trace.append(np.concatenate(p_values))

    def save(self, directory):
        np.savez(directory / 'mechanism_trace.npz', coordinate_indices=self.indices,
                 p_trace=np.stack(self.p_trace), r_trace=np.stack(self.r_trace),
                 s_trace=np.stack(self.s_trace))
        write_csv(directory / 'mechanism_metrics.csv', self.metrics)


def paper_approx_mf_distortion(r_trace, M, W):
    """Paper-style approximation ||W diag(r) M||_F²; not exact Adam Jacobian."""
    r = np.asarray(r_trace, dtype=np.float64)
    M, W = np.asarray(M, dtype=np.float64), np.asarray(W, dtype=np.float64)
    assert M.shape == W.shape == (r.shape[0], r.shape[0])
    H = (W.T @ W) * (M @ M.T)
    actual_sq = np.sum(r * (H @ r), axis=0)
    ideal_sq = np.mean(r ** 2, axis=0) * H.sum()
    assert np.all(actual_sq > 0) and np.all(ideal_sq > 0)
    ratio = np.sqrt(actual_sq / ideal_sq)
    return distortion_statistics(ratio, 'paper_approx')


PAPER_APPROX_METADATA = 'paper-style approximation; not exact Adam Jacobian'
FROZEN_V_METADATA = ('frozen-observed-v linearized noise operator; holds observed second-moment '
                     'trajectory fixed; excludes infinitesimal noise feedback into future v; '
                     'not exact Adam Jacobian')
CANCELLATION_BASELINE = (
    'K_iid_q = r_rms_q * L @ A @ (innovation_std_sum * I); '
    'r_rms_q = sqrt(mean((p_q/s_q)^2)); identity temporal transform, with the same '
    'trial privacy-calibrated innovation_std_sum and T * innovation_std_sum^2 '
    'pre-transform innovation energy. No new IID privacy calibration and no '
    'matching of post-transform output energy. Common lr/logical_batch_size cancels '
    'in ratios. Lower = more effective temporal noise cancellation.')


def distortion_statistics(ratio, prefix):
    logabs = np.abs(np.log(ratio))
    return {f'{prefix}_{key}': value for key, value in dict(
        ratio_p10=float(np.quantile(ratio, .1)), ratio_p50=float(np.median(ratio)),
        ratio_p90=float(np.quantile(ratio, .9)), logabs_mean=float(logabs.mean()),
        logabs_median=float(np.median(logabs))).items()}


def adam_first_moment_matrix(total_steps, beta1=.9):
    t, j = np.indices((total_steps, total_steps))
    lag = np.maximum(t - j, 0)
    return np.where(j <= t, (1 - beta1) * beta1 ** lag / (1 - beta1 ** (t + 1)), 0.)


def prefix_accumulation_matrix(total_steps):
    return np.tril(np.ones((total_steps, total_steps), dtype=np.float64))


def frozen_v_operator(p, s, M, beta1=.9):
    """Explicit L @ diag(p) @ A @ diag(1/s) @ actual saved M for one coordinate."""
    p, s, M = np.asarray(p), np.asarray(s), np.asarray(M)
    A = adam_first_moment_matrix(len(p), beta1)
    L = prefix_accumulation_matrix(len(p))
    return L @ np.diag(p) @ A @ np.diag(1 / s) @ M


def frozen_v_noise_norms(p_trace, s_trace, M, beta1=.9):
    """Exact Frobenius norms via Adam and prefix recurrences, O(Q*T²) work.

    Each row is an operator on all T innovations. Avoid Q dense matrix products
    and avoid storing [Q,T,T]; this is algebraically the specified frozen-v K.
    """
    p, s, M = (np.asarray(value, dtype=np.float64) for value in (p_trace, s_trace, M))
    assert p.shape == s.shape and p.ndim == 2
    T, Q = p.shape
    assert M.shape == (T, T)
    assert np.isfinite(M).all()
    assert np.isfinite(p).all() and np.isfinite(s).all() and (p > 0).all() and (s > 0).all()
    moment = np.zeros((Q, T))
    accumulated = np.zeros_like(moment)
    squared = np.zeros(Q)
    for t in range(T):
        moment *= beta1
        moment += (1 - beta1) * M[t][None, :] / s[t, :, None]
        accumulated += p[t, :, None] * moment / (1 - beta1 ** (t + 1))
        squared += np.sum(accumulated ** 2, axis=1)
    assert (squared > 0).all() and np.isfinite(squared).all()
    return np.sqrt(squared), np.sqrt(np.mean((p / s) ** 2, axis=0))


def frozen_v_diagnostics(p_trace, s_trace, M, innovation_std_sum, beta1=.9):
    """Frozen-v distortion and cancellation against a shared innovation baseline."""
    norm, r_rms = frozen_v_noise_norms(p_trace, s_trace, M, beta1)
    T = len(p_trace)
    LA = prefix_accumulation_matrix(T) @ adam_first_moment_matrix(T, beta1)
    ideal_norm = r_rms * np.linalg.norm(LA @ M, 'fro')
    # IID baseline retains the coordinate effective RMS and exactly the same
    # pre-transform privacy innovation energy T*sigma², with temporal D=I.
    # It is a mechanism diagnostic, not a separately calibrated IID trial.
    assert innovation_std_sum > 0
    iid_norm = r_rms * innovation_std_sum * np.linalg.norm(LA, 'fro')
    efficiency = (norm / np.sqrt(T)) / (iid_norm / np.sqrt(T))
    return dict(frozen_v_mf_distortion=distortion_statistics(norm / ideal_norm, 'frozen'),
                mf_cancellation_efficiency=dict(p10=float(np.quantile(efficiency, .1)),
                    p50=float(np.median(efficiency)), p90=float(np.quantile(efficiency, .9)),
                    mean=float(np.mean(efficiency))))


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
