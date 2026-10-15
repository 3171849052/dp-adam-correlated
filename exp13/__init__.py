"""Exp13: offline numerical experiments; all writable state stays here."""
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'exp13'
for key in ('TMPDIR', 'HF_HOME', 'HF_DATASETS_CACHE', 'TORCH_HOME',
            'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH'):
    path = BASE / 'runtime' / key.lower()
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
