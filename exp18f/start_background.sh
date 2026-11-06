#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
exec conda run --no-capture-output -n curve python -B -m exp18f.pipeline
