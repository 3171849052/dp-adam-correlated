#!/bin/bash
set -euo pipefail
cd /media/data/data/lt/dp-adam-correlated
export PYTHONDONTWRITEBYTECODE=1
export ARROW_DEFAULT_MEMORY_POOL=system
export MALLOC_CONF=background_thread:false
export PYTHONFAULTHANDLER=1
exec /home/longt29/miniconda3/envs/curve/bin/python -B -m exp16f.runtime.supervised launch
