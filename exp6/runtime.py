"""Offline libraries and every writable cache live under exp6."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'exp6'
RUNTIME = EXP / 'runtime'
RESULTS = EXP / 'results'


def configure():
    sys.dont_write_bytecode = True
    names = ('tmp', 'huggingface', 'torch', 'xdg', 'matplotlib', 'cuda', 'triton', 'inductor')
    for name in names:
        (RUNTIME / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(RUNTIME / 'huggingface'), TORCH_HOME=str(RUNTIME / 'torch'),
        XDG_CACHE_HOME=str(RUNTIME / 'xdg'), MPLCONFIGDIR=str(RUNTIME / 'matplotlib'),
        CUDA_CACHE_PATH=str(RUNTIME / 'cuda'), TRITON_CACHE_DIR=str(RUNTIME / 'triton'),
        TORCHINDUCTOR_CACHE_DIR=str(RUNTIME / 'inductor'), TMPDIR=str(RUNTIME / 'tmp'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
        PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import tempfile
    tempfile.tempdir = str(RUNTIME / 'tmp')


def output_path(path):
    path = Path(path).resolve()
    assert path.is_relative_to(EXP), f'Output outside exp6: {path}'
    return path


def require_curve():
    assert Path(sys.prefix).name == 'curve', 'Use conda run -n curve'


configure()
