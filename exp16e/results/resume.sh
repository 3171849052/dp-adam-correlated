#!/usr/bin/env bash
set -eu
cd /media/data/data/lt/dp-adam-correlated
exec /home/longt29/miniconda3/envs/curve/bin/python -B exp16e/runtime/supervised.py launch
