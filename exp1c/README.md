# exp1c

Independent CIFAR-100 BandInvMF tuning experiment. The model, augmentation,
GDP calibration, Toeplitz coefficients and logical update follow exp1.
The pretrained cache is copied locally and loaded offline; CIFAR-100 is read
from the repository's data/ with download=False. All experiment writes stay here.

The grid contains 3 Momentum learning rates and 9 Scale learning-rate ×
eps_scale combinations. Every trial uses seed 20261001, full FP32 fine-tuning,
50 physical examples, 20 microbatches, 1000 logical examples, 5 epochs,
clip 1, four bands, epsilon 8 and delta 1e-5. Participation and trajectory
length are derived from epochs and dataset size, never set in the config.

From the repository root, using the curve environment:

```bash
conda run --no-capture-output -n curve bash exp1c/run_sweep.sh
```

The launcher dynamically queues trials on physical GPUs 0,1,2,3, one process
per GPU. It skips completed trials, refuses incomplete existing directories,
and returns nonzero when a worker fails. It does not choose a winner.
Results and logs are under results/sweep/; results/sweep_summary.csv contains
all 12 rows, including pending/running/failed status when applicable.

Scale diagnostics use exact per-example norms from the same first backward
that measures the clipping geometry. The unscaled sampler reduces each layer's
per-example gradients before discarding them. Coordinates are measured once
at each logical step's start from frozen previous completed Adam vhat.
Epoch coordinate quantiles pool all trainable coordinates across those starts;
sample quantiles pool all examples in that epoch. These scalar arrays live
only in RAM until aggregation (about 2.2 GB per Scale epoch); no raw diagnostic
arrays are written. norm_stats.csv and scale_stats.csv contain aggregates only.
Sweep summary diagnostics and training metrics refer to the final epoch.
Marginal noise standard deviation is measured after division by logical batch
size, in gradient space for Momentum and scaled space for Scale.

Tests and isolated smoke artifacts are under results/. Smoke uses one complete
logical batch (20 physical microbatches) and 100 test images; privacy still
calibrates the full planned five-epoch trajectory. No full sweep has been run.

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp1c/cache/tmp" conda run --no-capture-output -n curve python -m pytest -c exp1c/pytest.ini exp1c/tests --basetemp=exp1c/results/pytest_tmp -q
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n curve python -m exp1c.verify_smoke
```
