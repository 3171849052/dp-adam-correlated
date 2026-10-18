# Exp14c

Run from repository root with the `curve` conda environment and `python -B`.
All generated files stay under exp14c. Historical Python kernels are executed in
exp14c module namespaces without importing historical package initializers.

Protocol: controlled AdaM3 (uncorrected m squared), beta1=.9, beta2=.999,
eps=1e-8, zero weight decay; no Scale or LR compensation. Per-run epsilon=8,
delta=1e-5, five epochs, logical batch=1000, physical CV=250/NLP=1000.

Commands:

```
conda run --no-capture-output -n curve python -B -m exp14c.tests
conda run --no-capture-output -n curve python -B -m exp14c.pipeline --smoke
conda run --no-capture-output -n curve python -B -m exp14c.start_background
```

The last command requires passing tests and all 12 two-step GPU smokes. It
creates a new process session, disconnects stdin, and redirects scheduler output
to results/scheduler.log. It records PID and exact command in background.json.
Workers are scheduled only on idle physical GPUs 0..3, at most one each.
No automatic retries. Failed trials are recorded; a failed stage prevents
selection/formal training. The scheduler finishes other jobs in that stage and
then records a failed pipeline status. Do not launch a second scheduler.

Automatic phases: 60 CV search -> 60 NLP search -> freeze eight winners ->
prepare official NLP tokens -> 80 formal runs -> strict audit -> report.
Search seed=20261101; formal seeds=20261111..20261120. Ranking uses last internal
validation accuracy descending, validation loss ascending, SHA trial ID ascending.
Official evaluations never enter selection. Full search evidence and code hashes
are bound by frozen_manifest.json and its SHA256 record.

NLP tokens are generated from data/sst2 parquet using the historical tokenizer
and checked exactly against Exp9 tokens. Historical immutable split files and
MF matrices are read-only references. All pretrained weights load from cache/.
The full historical experiment/data/cache file size+mtime snapshot is verified
before launch and after completion. Kernel source and run artifact SHA256 hashes
are retained. Matrix construction and noise innovation streams are checked
against Exp9. Formal runs pair exactly on initialization, data order, RNG trace,
and matrix hashes; noise amplitude is recalibrated for the selected C.

Read results/pipeline_status.json, *_status.json and scheduler.jsonl for progress.
Each trial stores config, per-epoch and per-step diagnostics, checkpoint, summary,
process metadata and train.log. results/final_report.md is generated only after
all 200 full runs complete; statistics use sample SD and Student-t 95% CIs, with
paired seed differences against Exp9 Adam in percentage points.

Privacy is per training run only, not a joint epsilon=8 guarantee for tuning or
repetition. Historical official evaluations have been viewed; no blind-test claim.
