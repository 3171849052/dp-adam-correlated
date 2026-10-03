#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/exp1e/cache/tmp"
mkdir -p "$TMPDIR"
exec python -u -m exp1e.sweep "$@"
