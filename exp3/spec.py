"""Strict machine-readable trial specification, without a search grid."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
EXP3 = ROOT / 'exp3'
METHODS = ('nonprivate_hybrid', 'iid_dp_hybrid', 'mf_muon_standard',
           'mf_muon_normscale', 'mf_muon_spectralscale')
TRIAL_FILES = ('config.yaml', 'train.log', 'metrics.csv', 'summary.json',
               'train_order.npy', 'final.pt', 'matrices.npz', 'diagnostics.csv', 'diagnostics.json',
               'update_statistics.csv', 'update_statistics.json', 'muon_trajectory.pt', 'frozen_trajectory_muon_mf.json')
IMPLEMENTATION_VERSION = 'active_spectrum_v2_update_space_frozen_trajectory'


@dataclass(frozen=True)
class TrialSpec:
    method: str
    result_dir: str
    seed: int = 20261001
    muon_lr: float = .01
    adam_lr: float = .0005
    max_grad_norm: float = 100.
    lambda_parallel: float = 1.
    kappa: float = 4.
    rho: float = .1
    smoke: bool = False
    diagnostic_interval: int = 25
    diagnostic_probes: int = 4

    def __post_init__(self):
        for name in ('muon_lr', 'adam_lr', 'max_grad_norm', 'lambda_parallel', 'kappa', 'rho'):
            object.__setattr__(self, name, float(getattr(self, name)))
        assert self.method in METHODS
        assert type(self.seed) is int and 0 <= self.seed < 2**32 - 3
        assert type(self.smoke) is bool
        assert type(self.diagnostic_interval) is int and self.diagnostic_interval > 0
        assert type(self.diagnostic_probes) is int and self.diagnostic_probes > 0
        for value in (self.muon_lr, self.adam_lr, self.max_grad_norm, self.lambda_parallel, self.rho):
            assert math.isfinite(value) and value > 0
        assert math.isfinite(self.kappa) and self.kappa >= 1
        assert self.directory.is_relative_to(EXP3) and self.directory != EXP3

    @property
    def directory(self):
        path = Path(self.result_dir)
        return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()

    def mapping(self):
        values = asdict(self)
        values['result_dir'] = str(self.directory.relative_to(ROOT))
        return values

    def fingerprint(self):
        return hashlib.sha256(json.dumps(dict(version=IMPLEMENTATION_VERSION, spec=self.mapping()), sort_keys=True).encode()).hexdigest()


def load_spec(path):
    return TrialSpec(**json.loads(Path(path).read_text()))


def read_completed(spec):
    directory = spec.directory
    for name in TRIAL_FILES:
        if not (directory / name).is_file() or not (directory / name).stat().st_size:
            raise RuntimeError(f'Incomplete trial: {directory}; missing or empty {name}')
    summary = json.loads((directory / 'summary.json').read_text())
    if summary['status'] != 'completed' or summary['spec_sha256'] != spec.fingerprint():
        raise RuntimeError(f'Existing trial conflicts with requested spec: {directory}')
    return summary
