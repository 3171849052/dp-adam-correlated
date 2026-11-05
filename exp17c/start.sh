#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/exp17c/runtime/vendor${PYTHONPATH:+:$PYTHONPATH}"
exec /home/longt29/miniconda3/envs/curve/bin/python -B -m exp17c.start_background
