"""Exp2 GDP and absolute-Gram fixed participation; no amplification."""
from exp2.privacy import calibrate, epsilon_from_mu, fixed_epoch_sensitivity
from exp6.config import FIXED


def calibration(strategy, trial):
    return calibrate(strategy, {'privacy': dict(epsilon=FIXED['epsilon'], delta=FIXED['delta'],
        k=FIXED['epochs'], b_participation=FIXED['steps_per_epoch'],
        max_grad_norm=trial.C, adjacency=FIXED['adjacency'])})


def spent(strategy, steps, trial, privacy):
    sensitivity = fixed_epoch_sensitivity(strategy, 5, 50, steps=steps)
    mu = trial.C * sensitivity / privacy['innovation_std_sum']
    return dict(mu=mu, epsilon=epsilon_from_mu(mu, FIXED['delta']), delta=FIXED['delta'])
