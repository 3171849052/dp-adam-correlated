#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
methods=(adam dp_adam dp_adam_bandinvmf_momentum dp_adam_bandinvmf_scale)
pids=()
result_base=exp1/results
if [[ "${1:-}" == "--smoke" ]]; then
    result_base=exp1/results/smoke
fi
for gpu in 0 1 2 3; do
    method="${methods[$gpu]}"
    mkdir -p "$result_base/$method"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        python -u -m exp1.train --method "$method" "$@" \
        > "$result_base/$method/train.log" 2>&1 &
    pids+=("$!")
done
status=0
for i in 0 1 2 3; do
    if wait "${pids[$i]}"; then
        echo "${methods[$i]} completed"
    else
        child_status=$?
        echo "${methods[$i]} failed (exit $child_status); see $result_base/${methods[$i]}/train.log" >&2
        status=1
    fi
done
exit "$status"
