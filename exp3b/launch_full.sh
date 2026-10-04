#!/usr/bin/env bash
set -euo pipefail

experiment_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$experiment_root"
source /home/longt29/miniconda3/etc/profile.d/conda.sh
conda activate curve
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="$experiment_root/exp3b/runtime/tmp"

python - <<'PY'
from pathlib import Path
import os
assert Path(os.environ['CONDA_PREFIX']).name == 'curve'
assert not Path('exp3b/results/full_experiment.pid').exists(), 'Background PID file already exists; inspect it before another launch'
assert Path('exp3b/runtime/tmp').is_dir()
PY

nohup setsid python -u -m exp3b.experiment --gpus 0,1,2,3 --data-root data --cache-root cache > exp3b/results/full_experiment.log 2>&1 < /dev/null &
experiment_pid=$!
printf '%s\n' "$experiment_pid" > exp3b/results/full_experiment.pid
python - "$experiment_pid" <<'PY'
from datetime import datetime
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
from exp3b.spec import write_json

pid = int(sys.argv[1])
assert Path(f'/proc/{pid}').is_dir(), 'Background process exited; inspect full_experiment.log'
write_json('exp3b/results/background_launch.json', dict(
    status='launched', pid=pid, started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    conda_environment='curve', python=sys.executable, conda_prefix=os.environ['CONDA_PREFIX'],
    gpus=[0, 1, 2, 3], max_concurrency=4, data_root='data', cache_root='cache',
    command=[sys.executable, '-u', '-m', 'exp3b.experiment', '--gpus', '0,1,2,3',
             '--data-root', 'data', '--cache-root', 'cache'],
    log='exp3b/results/full_experiment.log', user_requested_background_launch=True))
print(f'Launched complete Exp3b experiment in background: PID={pid}; log=exp3b/results/full_experiment.log')
PY
