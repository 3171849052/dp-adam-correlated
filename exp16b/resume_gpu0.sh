#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  ""|--check) ;;
  *) echo "Usage: $0 [--check]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
experiment_python=/home/longt29/miniconda3/envs/curve/bin/python

"$experiment_python" -B - <<'PY'
import fcntl
from collections import Counter
from exp16b import RESULTS
from exp16b.search import coarse_jobs, freeze
from exp16b.stage1 import verify_acceptance
from exp16b.scheduler import verify, gpu_processes

with (RESULTS / 'pipeline.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_acceptance()
    freeze()
    counts = Counter()
    for trial in coarse_jobs():
        if (trial.output / 'summary.json').exists():
            counts[verify(trial)['status']] += 1
        elif trial.output.exists():
            raise RuntimeError(f'Unreviewed partial trial: {trial.output}')
        else:
            counts['pending'] += 1
    print(f'Restart preflight passed: {dict(counts)}', flush=True)
    print(f'Physical GPU 0 compute PIDs: {sorted(gpu_processes()[0])}', flush=True)
    print('Completed trials will be reused; interrupted trial will restart from step 0.', flush=True)
PY

if [[ "${1:-}" == --check ]]; then
  exit 0
fi
exec "$experiment_python" -B -m exp16b.start_background
