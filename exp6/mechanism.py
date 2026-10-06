"""One vector innovation per logical step; shared Gaussian seeds across cells."""
import hashlib
import numpy as np
from exp2.bandinvmf import build_matrices, materialize, BandInvMFNoise
from exp6.config import FIXED


def matrices(trial):
    return build_matrices(trial.noise, FIXED['total_steps'], FIXED['num_bands'], FIXED['beta1'])


class TemporalNoise(BandInvMFNoise):
    def __init__(self, named_parameters, coefficients, innovation_std, seed):
        self.names = list(named_parameters)
        # Retain standardized innovations. Scale only the output; this also permits
        # exact cross-cell audits when the calibrated noise std differs.
        super().__init__(named_parameters.values(), coefficients, 1., FIXED['total_steps'], seed)
        self.calibrated_std = float(innovation_std)
        self.innovation_digests = []

    def draw(self):
        output = self.next()
        innovations = self.history[0] if self.history else output  # IID d[0] = 1 exactly
        digest = hashlib.sha256()
        for z in innovations:
            digest.update(z.cpu().numpy().tobytes())
        self.innovation_digests.append(digest.hexdigest())
        return {n: z * self.calibrated_std for n, z in zip(self.names, output)}

    def marginal_std(self, step):
        return self.calibrated_std * float(np.linalg.norm(self.coefficients[:min(step+1, len(self.coefficients))]))
