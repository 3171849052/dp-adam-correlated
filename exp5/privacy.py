"""Exp2 sparse participation accounting, adapted to replace-one averages."""
import numpy as np
from exp2.privacy import target_mu, epsilon_from_mu, fixed_epoch_sensitivity
from exp2.bandinvmf import build_matrices, materialize
from exp5.config import FIXED, METHODS


def build(method, C):
    num_bands = FIXED['num_bands']
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    T, beta = FIXED['total_steps'], FIXED['beta']
    noise_kind = 'iid' if method == METHODS[0] else 'momentum_bandinvmf'
    coefficients, strategy, _ = build_matrices(noise_kind, T, num_bands, beta)
    # Exact exp2 momentum workload: prefix composed with unnormalized momentum.
    workload = np.cumsum(beta ** np.arange(T))
    mu = target_mu(FIXED['epsilon'], FIXED['delta'])
    per_query = 2 * C / FIXED['logical_batch_size']
    multiplier = fixed_epoch_sensitivity(strategy, FIXED['epochs'], FIXED['steps_per_epoch'])
    sensitivity = per_query * multiplier
    metadata = dict(epsilon=FIXED['epsilon'], delta=FIXED['delta'], mu=mu,
                    adjacency='replace_one', per_query_sensitivity=per_query, per_step_sensitivity=per_query,
                    sensitivity=sensitivity, strategy_sensitivity=multiplier,
                    innovation_std=sensitivity / mu, participation='fixed_epoch_sparse',
                    epochs=5, participation_spacing=50, direct_participations=5,
                    sampling_amplification=False, workload='momentum', beta=beta,
                    num_bands=num_bands if method == METHODS[1] else 1,
                    noising_coefficients=coefficients.tolist(),
                    strategy='inverse_lower_triangular_toeplitz',
                    accountant_source='exp2.privacy.fixed_epoch_sensitivity',
                    workload_source='exp2.bandinvmf.build_matrices(momentum_bandinvmf)',
                    noise_space='logical_averaged_gradient')
    return coefficients, strategy, workload, metadata


def spent(strategy, metadata, steps):
    multiplier = fixed_epoch_sensitivity(strategy, 5, 50, steps=steps)
    mu = metadata['per_query_sensitivity'] * multiplier / metadata['innovation_std']
    return dict(mu=mu, epsilon=epsilon_from_mu(mu, FIXED['delta']), delta=FIXED['delta'])
