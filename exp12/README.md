# Exp12 CV BC2-Naive BandInvMF

From the project root, run:

```bash
conda run --no-capture-output -n curve python -B -m exp12.run --gpus 0 1 2
```

This single command validates both horizons, GDP calibration, exact clipping,
Adam accumulation, offline pretrained mapping and Exp9 tie-breaking; runs short
GPU smoke on each device; then immediately executes 20 grid trainings, 4 Top-2
rechecks and 10 frozen-config formal trainings. Every full training has 5 epochs.
No automatic retries or incomplete-record reuse. A failed execution retains its
logs and records and must be inspected before a deliberate restart.

Only the matrix-optimization workload changes:

W[t,j] = sum(s=j..t) beta1^(s-j) sqrt(1-beta2^s)/(1-beta1^s), j <= t.

beta1=0.9, beta2=0.999, no global (1-beta1). Four-tap D=[1,d1,d2,d3].
Objective Sens(D^-1)^2 ||W D||_F^2/T. Identical Exp9 optimizer and fixed starts.
T=225 for selection; T=250 for full 50000-example formal training.

The Exp9 non-Scale Momentum-Bias method identifier is retained for canonical
trial IDs and identical tie ordering; the Exp12 directory and workload metadata
identify the changed experiment. Historical files are never imported as Python
packages, preventing Exp9 runtime writes. Copied kernels use local imports.
Data split, augmentation, initialization, optimizer, FP32 arithmetic, RNG,
privacy calibration and five-epoch evaluation semantics come directly from Exp9.
NLP and Scale experiments are excluded by Trial validation.

Formal seeds: 20261111 through 20261120. Test results never affect selection.
Baseline pairing requires verified historical artifacts and matching per-seed
initialization, classifier, training order and augmentation RNG traces. Invalid
historical records cannot contribute paired results.

All new files remain in exp12/. Outputs: results/report.md, grid_results.csv,
recheck_results.csv, final_results.csv, matrix_comparison.csv, statistics.json,
paired_differences.csv, privacy_audit.csv, compute_summary.csv, figures,
GPU telemetry, checkpoints and frozen/result SHA-256 manifests.
