# Exp14a

Run from the repository root in conda `curve`, always with `python -B`.

```
conda run --no-capture-output -n curve python -B -m exp14a.tests
conda run --no-capture-output -n curve python -B -m exp14a.launch --smoke
conda run --no-capture-output -n curve python -B -m exp14a.start_background
```

The last command checks test/smoke evidence and launches a detached session. The actual detached launch
and PID are recorded in results/background.json. Four FIFO slots map to physical
GPUs 0,1,2,3, each with exactly one worker. All eight formal jobs are submitted;
no retry, hyperparameter search, Scale, or historical training is performed.

Necessary Exp9 kernels are copied with local imports to avoid Exp9 package
initialization writing runtime caches. Original source hashes are recorded in
results/source_provenance.json. Data loaders read immutable Exp9 split/token
artifacts derived from local data/. Models read local cache/ only. Matrices are
loaded directly from the paired Exp9 formal artifacts; tests independently
reconstruct and check every entry, calibration and full-horizon noise stream.

AdaM3 squares the uncorrected momentum, with separate Adam bias corrections,
betas (.9,.999), epsilon 1e-8 and no weight decay. Tests compare 30 float64 steps
with the explicit recurrence and verify exact serialized optimizer continuation.
Training uses the original seed, loading, augmentation, dropout, clipping and
independent DP random stream. No RNG calls are added by diagnostics.

Each trial saves config.json, matrices.npz, train_order.npy, metrics.csv (each
epoch), mechanism_metrics.csv (each step), optimizer_metrics.jsonl (steps 1–20,
then every 10, plus anomalies), checkpoint.pt, summary.json, process.json and
train.log. The final checkpoint contains model and optimizer states; this does
not implement automatic full-training resumption. Every step checks finite
model and moment states. Scheduler states and commands are persisted in
results/scheduler_status.json and results/scheduler.jsonl. After all workers exit,
report.py writes summary.csv, comparison.csv, training_curves.png and
final_report.md, including failed statuses if any. Successful runs also verify
paired initialization, data and augmentation/dropout traces against Exp9.

One-seed differences are descriptive only. GPU timing includes diagnostic costs.
