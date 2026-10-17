# Exp14b

Run from the repository root with conda `curve` and `python -B`:

```
conda run --no-capture-output -n curve python -B -m exp14b.tests
conda run --no-capture-output -n curve python -B -m exp14b.pipeline --smoke
conda run --no-capture-output -n curve python -B -m exp14b.start_background
```

The last command detaches the scheduler with a new session, null stdin and file
logs. It runs exactly 36 full search trainings, then four selected confirmation
trainings and report generation, with no retry. Physical GPUs 0–3 each have at
most one worker; the next queued job takes the next free GPU. Search order
interleaves mechanisms; trial numbers remain fixed in method/lr/eps order.

All new code is here and all runtime/output files are under results/, including
cache directories. Historical code is read through reuse.py without importing
historical package initializers. Its imports are redirected locally. The only
training-body changes are the configurable optimizer epsilon and removal of the
historical fixed-lr assertion. The optimizer itself is unchanged. Historical
sources and local modules are hashed in source_provenance.json and trial configs.

Search uses Exp9's saved internal 45,000/5,000 split, 225-step matrices and the
fixed seed 20261111. Confirmation uses 50,000/10,000 train/test and 250-step
matrices. Matrices are read from historical artifacts and independently rebuilt
in tests. Same preprocessing, clipping, initialization and independent noise RNG.
No downloads, historical training or historical writes are performed.

Each training saves config.json, train.log, metrics.csv, mechanism_metrics.csv,
optimizer_metrics.jsonl, matrices.npz, train_order.npy, checkpoint.pt and summary.json.
Optimizer diagnostics include steps 1–20 and every tenth later step, plus anomalies;
finite model and moment checks run every logical step. Smoke consists of two
logical steps per mechanism at the smallest lr and eps, without held-out evaluation.

Selection orders final internal accuracy descending, final validation loss
ascending, then fixed trial number ascending. best_configs.json is written before
any confirmation worker starts. Search and confirmation status/commands/PIDs are
in search_status.json and confirmation_status.json, with append-only event logs.
The completed report includes rankings, comparisons, heatmaps and curves. Test
accuracy is never used for hyperparameter selection. Single-seed comparisons are
descriptive, and epsilon=8 is a per-training budget, not a composed search claim.
