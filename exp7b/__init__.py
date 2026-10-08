"""Exp7b: seven methods and paired final seeds; mathematical kernels are imported from Exp2."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'exp7b'
def runtime():
    for key, value in {'HF_HOME':'cache/huggingface', 'TORCH_HOME':'cache/torch',
                       'XDG_CACHE_HOME':'cache', 'TMPDIR':'runtime/tmp',
                       'MPLCONFIGDIR':'cache/matplotlib'}.items():
        path = BASE / value
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                      JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2')
runtime()
