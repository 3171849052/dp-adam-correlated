# Exp9b: four-band BandMF

Exp9 protocol and all 643 trainings are reproduced with S=I+c1L+c2L²+c3L³, D=S⁻¹. IID is rerun, and no historical training is reused. Full non-Toeplitz Momentum-Bias workload, fixed participation GDP, clipping, Adam and Scale are preserved. IIR retains the last three outputs.

All writes stay under exp9b; data/ and cache/ are offline and read-only. Physical GPUs 0,1,2 participate; GPU 3 is reserved. CV capacity is one, NLP two after single-worker and concurrent GPU smokes.

```bash
conda run --no-capture-output -n curve python -B -m exp9b.stage1 --gpus 0 1 2
nohup conda run --no-capture-output -n curve python -B -m exp9b.stage2 --gpus 0 1 2 --cv-per-gpu 1 --nlp-per-gpu 2 > exp9b/results/full_run.log 2>&1 < /dev/null &
```

The background script executes 321 search, 56 rechecks, immutable 14-winner freeze, 140 formal, 126 epsilon sweeps, audit and report. No retry or batch changes. Failures stop dependent stages. The launcher itself retains per-worker logs and scheduling telemetry; no external monitor is needed.

Source hashes: kernel_provenance.json and platform_validation.json. Config/matrix/checkpoint/metrics hashes: trial summaries. Frozen matrix and selection hashes: frozen_manifest.json. All finished result artifacts: results_manifest.json. Report: results/final_report.md; paired historical differences with t 95% intervals: exp9_paired_summary.csv; full matrix errors, sensitivity, final/final-epoch noise variance: matrix_geometry.csv. Hyperparameters are independently selected for each experiment; historical comparison is same method and seed.
