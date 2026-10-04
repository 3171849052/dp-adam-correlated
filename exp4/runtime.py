"""Set offline and write locations before importing numerical/model libraries."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'exp4'
RUNTIME = EXP / 'runtime'
for name in ('tmp', 'huggingface', 'torch', 'xdg', 'matplotlib'):
    (RUNTIME / name).mkdir(parents=True, exist_ok=True)
os.environ.update(
    HF_HOME=str(RUNTIME / 'huggingface'), TORCH_HOME=str(RUNTIME / 'torch'),
    XDG_CACHE_HOME=str(RUNTIME / 'xdg'), MPLCONFIGDIR=str(RUNTIME / 'matplotlib'),
    TMPDIR=str(RUNTIME / 'tmp'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1',
    JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true', CUBLAS_WORKSPACE_CONFIG=':4096:8',
)


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(EXP):
        raise ValueError(f'Output must be inside {EXP}: {path}')
    return path
