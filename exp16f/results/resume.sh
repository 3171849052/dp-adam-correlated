#!/bin/bash
set -euo pipefail
cd /media/data/data/lt/dp-adam-correlated
export PYTHONDONTWRITEBYTECODE=1
exec /home/longt29/miniconda3/envs/curve/bin/python -B -m exp16f.runtime.recover launch
