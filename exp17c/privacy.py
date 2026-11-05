"""Independent Poisson Gaussian mechanism; accounting delegated to Opacus."""
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


def calibrate(stage):
    if stage not in ('search', 'final'):
        raise ValueError(stage)
    size, steps = (45000, 225) if stage == 'search' else (50000, 250)
    q = 1000 / size
    sigma = get_noise_multiplier(target_epsilon=8.0, target_delta=1e-5,
                                 sample_rate=q, steps=steps, accountant='rdp',
                                 epsilon_tolerance=0.001)
    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=sigma, sample_rate=q)
    epsilon = accountant.get_epsilon(delta=1e-5)
    if not 7.999 <= epsilon <= 8.0:
        raise ValueError(f'Calibration mismatch: {epsilon}')
    return dict(stage=stage, population=size, steps=steps, sample_rate=q,
                noise_multiplier=sigma, actual_epsilon=epsilon, delta=1e-5,
                nu_squared=(sigma * 30 / 1000) ** 2)
