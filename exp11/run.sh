#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/exp11/runtime/tmpdir"
export XDG_CACHE_HOME="$PWD/exp11/runtime/xdg_cache_home"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" exp11/results
exec conda run --no-capture-output -n curve python -B -m exp11.run --gpus 0 1 2 3 "$@"
