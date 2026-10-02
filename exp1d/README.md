# exp1d: Momentum learning rate and Scale clipping sweep

Standalone copy of the verified exp1c implementation. Only existing repository
`data/` CIFAR-100 is read (`download=False`); pretrained weights are copied into
`exp1d/cache/huggingface/` and loaded offline. Missing dependencies, weights,
data or GPUs cause failure. All runtime artifacts stay under `exp1d/`.

Fixed: pretrained ViT-Tiny, 100 classes, full fine-tuning, FP32, five epochs,
logical batch 1000 / physical batch 250 / accumulation 4, Adam (.9,.999),
eps 1e-8, weight decay 0, epsilon 8, delta 1e-5, four bands. Participation
is derived from epochs and dataset size: k=5, b=50, T=250. No amplification;
add/remove zero-out adjacency; whole-trajectory fixed-epoch GDP.

The four Momentum trials use lr {2e-3,4e-3,5e-3,7e-3}, clip 1. The three
Scale trials use lr 1e-3, eps_scale .1, scaled clip {10,100,1000}. Every trial
runs calibration using its own clip, before training. Every physical batch
uses exact per-example clipping; each logical batch emits correlated noise
once and updates Adam once. Scale clipping and diagnostics use the frozen
previous completed vhat, including vhat_0=0. Only epoch aggregates are saved.

All trials reset the same seed, pretrained/classifier initialization,
fixed training permutation and DataLoader augmentation RNG setup. Noise has
an independent seeded generator. Each trial saves config.yaml, train.log,
metrics.csv, summary.json, train_order.npy, final.pt and matrices.npz. Scale
also saves norm_stats.csv and scale_stats.csv. The launcher writes
results/sweep_summary.csv without selecting a best trial.

Run from the repository root:

```bash
conda run --no-capture-output -n curve bash exp1d/run_sweep.sh
```

The launcher starts four Momentum trials on GPUs 0–3, then queues Scale trials
on freed GPUs. Each process uses one GPU. Any failed trial makes the launcher
exit nonzero after the queue finishes. If `results/sweep` or its summary already
exists, a new run is created under `results/runs/<timestamp>_<unique suffix>/`,
with its own `sweep/` and `sweep_summary.csv`. The launcher prints both paths.
Existing results are preserved; each invocation starts all seven trials afresh.

Unit tests (including mock scheduler success/failure tests):

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp1d/cache/tmp" conda run --no-capture-output -n curve python -m pytest -c exp1d/pytest.ini exp1d/tests --basetemp=exp1d/results/pytest_tmp
```

GPU smoke runs use `exp1d.train --smoke`, which trains only one logical batch
(4 full-size microbatches) and evaluates 100 examples while retaining the full
250-step privacy calibration. They do not constitute a complete experiment.
