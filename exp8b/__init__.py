"""BERT-Tiny / SST-2 compute configuration experiment."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'exp8b'
RESULTS = BASE / 'results'
for name, path in {'TMPDIR': BASE/'runtime', 'HF_HOME': ROOT/'cache/huggingface',
                   'HF_DATASETS_CACHE': ROOT/'data/huggingface',
                   'XDG_CACHE_HOME': BASE/'runtime/cache'}.items():
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(path)
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', JAX_PLATFORMS='cpu',
                  JAX_ENABLE_X64='true', HF_HUB_DISABLE_XET='1',
                  TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
