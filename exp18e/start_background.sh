#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/longt29/miniconda3/etc/profile.d/conda.sh
conda activate curve
export PYTHONDONTWRITEBYTECODE=1
python -B -m exp18e.start_background
