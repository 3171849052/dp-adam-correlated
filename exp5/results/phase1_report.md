# Exp5 Phase 1 validation

17 tests passed in curve. Smoke completed on GPUs 0,1,2. Chunk size: 50.

| method | seed | allocated MiB | reserved MiB | logical step seconds |
|---|---:|---:|---:|---:|
| dp-coordnorm-sgdm-iid | 20261001 | 4950.65 | 5768.00 | 17.406 |
| dp-coordnorm-sgdm-bandinvmf | 20261001 | 4950.65 | 5770.00 | 14.557 |
| dp-coordnorm-sgdm-iid | 20261002 | 4950.65 | 5768.00 | 17.410 |

Logical-step timings include initial data-loader startup. Full-step protocol uses ten physical batches of 100.

Probe quantiles: {"p10": 0.0008230606833437359, "p25": 0.002648620499655345, "p50": 0.007644036157466592, "p75": 0.018593019388332428, "p90": 0.03763713104171447}.
The probe is nonprivate and excluded from the fixed-configuration per-run privacy calibration.

Steady-state logical-step median from the five completed Stage 1 trials: 2.039 seconds (epoch loader startup excluded).
