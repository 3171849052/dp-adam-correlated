"""Conservative full-participation Gaussian/GDP calibration, no sampling.

For adaptive bounded UC queries, each conditional whitened mean difference
is bounded by 2R sum_j |C[t,j]|. Compose these Gaussian bounds over ALL rows.
No sparse epoch participation or division by minibatch size is used.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr


def gdp_delta(mu, epsilon):
    a, b = -epsilon / mu + mu / 2, -epsilon / mu - mu / 2
    log_a = log_ndtr(a)
    return float(np.exp(log_a) * (-np.expm1(epsilon + log_ndtr(b) - log_a)))


def target_mu(epsilon, delta):
    return brentq(lambda mu: gdp_delta(mu, epsilon) - delta, 1e-6, 100.0)


def epsilon_from_mu(mu, delta):
    if mu == 0 or gdp_delta(mu, 0) <= delta:
        return 0.0
    return brentq(lambda eps: gdp_delta(mu, eps) - delta, 0.0, 1000.0)


def full_temporal_sensitivity(strategy, steps=None):
    """Dimensionless row-sum bound, valid also for adaptive query directions.

    For our nonnegative C this equals sqrt(sum(C.T @ C)), the tight aligned
    full-participation bound. For C=I it is sqrt(T).
    """
    c = np.asarray(strategy, dtype=np.float64)
    assert c.ndim == 2 and c.shape[0] == c.shape[1]
    rows = c if steps is None else c[:steps]
    return float(np.linalg.norm(np.abs(rows).sum(axis=1)))


def calibrate(strategy, update_clip_norm, epsilon=8.0, delta=1e-5):
    mu = target_mu(epsilon, delta)
    temporal = full_temporal_sensitivity(strategy)
    sensitivity = 2 * update_clip_norm * temporal
    return dict(target_epsilon=epsilon, delta=delta, target_mu=mu,
                total_steps=len(strategy), per_step_sensitivity=2 * update_clip_norm,
                temporal_sensitivity=temporal, trajectory_sensitivity=sensitivity,
                innovation_std=sensitivity / mu, sampling_amplification=False,
                participation='full_temporal', adjacency='replace_one',
                accountant='conditional_gaussian_gdp_row_absolute_sum',
                noise_space='update_direction', batch_size_division=False)
