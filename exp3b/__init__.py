"""Offline Exp3 continuation and fixed-signal optimizer cancellation."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'exp3b'
sys.dont_write_bytecode = True


def offline_runtime():
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    for key, value in {'HF_HOME': 'huggingface', 'TORCH_HOME': 'torch',
                       'XDG_CACHE_HOME': 'xdg', 'MPLCONFIGDIR': 'matplotlib',
                       'TRITON_CACHE_DIR': 'triton', 'TMPDIR': 'tmp'}.items():
        os.environ[key] = str(BASE / 'runtime' / value)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['JAX_ENABLE_X64'] = 'true'


offline_runtime()
