"""Strict local-only run specification and output validation."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from exp3b import BASE, ROOT

HYBRID = ('nonprivate_hybrid', 'iid_dp_hybrid', 'mf_muon_standard',
          'mf_muon_normscale', 'mf_muon_spectralscale')
ADAM = ('momentum_standard', 'momentum_scale')
PARAMETERS = ('muon_lr', 'adam_lr', 'max_grad_norm', 'lambda_parallel', 'kappa', 'rho')
VERSION = 'exp3b_fixed_signal_complete_update_v1'


def output_path(path):
    path = (ROOT / path).resolve()
    assert path.is_relative_to(BASE) and path != BASE, path
    return path


def write_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@dataclass(frozen=True)
class RunSpec:
    method: str
    result_dir: str
    seed: int = 20261001
    muon_lr: float = .006
    adam_lr: float = .012
    max_grad_norm: float = 30.
    lambda_parallel: float = 1.
    kappa: float = 4.
    rho: float = .1
    lr: float = .005
    eps_scale: float = .1
    capture: bool = False
    smoke: bool = False
    data_root: str = 'data'
    cache_root: str = 'cache'

    def __post_init__(self):
        assert self.method in HYBRID + ADAM
        assert type(self.seed) is int and 0 <= self.seed < 2**32 - 3
        assert type(self.capture) is bool and type(self.smoke) is bool
        assert not self.capture or self.method in ADAM + ('mf_muon_standard',)
        for key in PARAMETERS + ('lr', 'eps_scale'):
            assert math.isfinite(getattr(self, key)) and getattr(self, key) > 0
        assert self.kappa >= 1 and self.eps_scale == .1
        assert (ROOT / self.data_root).resolve() == ROOT / 'data'
        assert (ROOT / self.cache_root).resolve() == ROOT / 'cache'
        output_path(self.result_dir)

    @property
    def directory(self):
        return output_path(self.result_dir)

    def mapping(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(dict(version=VERSION, spec=self.mapping()),
                                         sort_keys=True).encode()).hexdigest()


def read_completed(spec):
    result = json.loads((spec.directory / 'summary.json').read_text())
    assert result['status'] == 'completed'
    assert result['spec_sha256'] == spec.fingerprint()
    assert result['optimizer_steps'] == (2 if spec.smoke else 250)
    assert math.isfinite(result['final_test_top1'])
    for file in ('metrics.csv', 'config.json', 'train.log', 'matrices.npz', 'final.pt'):
        assert (spec.directory / file).stat().st_size > 0
    if spec.capture:
        manifest = json.loads((spec.directory / 'signals/manifest.json').read_text())
        assert manifest['steps'] == result['optimizer_steps']
        assert len(manifest['files']) == manifest['steps']
        for entry in manifest['files']:
            assert (spec.directory / 'signals' / entry['file']).stat().st_size > 0
    return result
