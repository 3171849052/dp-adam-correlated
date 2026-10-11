"""Experiment-local runtime; raw data and pretrained weights are read-only."""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'exp10'
RESULTS = BASE / 'results'
for key in ('TMPDIR', 'HF_HOME', 'HF_DATASETS_CACHE', 'TORCH_HOME',
            'XDG_CACHE_HOME', 'MPLCONFIGDIR'):
    path = BASE / 'runtime' / key.lower()
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                  JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
