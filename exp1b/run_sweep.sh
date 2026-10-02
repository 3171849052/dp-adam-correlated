#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/exp1b/cache/tmp"
mkdir -p "$TMPDIR"
exec python -u -m exp1b.sweep "$@"
