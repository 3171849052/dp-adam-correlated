#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/exp2/cache/tmp"
export XDG_CACHE_HOME="$PWD/exp2/cache/xdg"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
exec python -u -m exp2.sweep "$@"
