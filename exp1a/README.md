# Adam / DP-Adam diagnostics

Independent copy of exp1's pretrained ViT mapping, transforms, FP32 Adam and
fixed-epoch GDP calibration. All parameters train. CIFAR-100 is read only from
repository `data/`, with `download=False`. Pretrained assets are copied into
`exp1a/cache`; Hugging Face runs offline and missing assets fail directly.

`config.yaml` contains epochs and batch sizes, never handwritten k/b/T.
The default derives (k,b,T)=(5,50,250), with 20 physical batches of 50 per
logical batch of 1000. The original fixed permutation is reused each epoch.
Training and evaluation loaders use separate seeded RNGs, identical for all
trials. IID Gaussian noise uses its own generator and cannot change augmentation.

The grid contains 3 Adam, 12 DP-Adam, and 12 clipped Adam without noise trials.
There is no adaptive selection, test-accuracy sorting, or best-trial output.
Clipped Adam and DP-Adam share the same clipping and accumulation functions;
only DP-Adam supplies a noise object. Noiseless privacy calibration is null.

Calibration uses the identity strategy and the whole fixed-epoch trajectory:
mu solves delta(mu,epsilon)=1e-5 for epsilon=8, sensitivity=sqrt(k), and Gaussian
standard deviation on clipped sums is C*sqrt(k)/mu. Metrics and sweep summary
`noise_std` use the averaged gradient space (sum standard deviation / 1000).
`innovation_std_sum` is also recorded. No sampling amplification or Opacus
sampling accountant is used. Epoch epsilon is computed from the trajectory
prefix; the fifth epoch reaches the target.

Each clipped microbatch measures exact per-parameter per-example norms before
clipping. Head/backbone/full norms use sums of these same squared norms.
RAM retains epoch norms for exact quantiles; only epoch aggregates reach disk.
`norm_stats.csv` has one row per epoch and norm group. Each group's clip_fraction
reports the fraction exceeding C; the full-model value is the clipping rate.
`summary.json` includes all last-epoch norm aggregates. Adam's norm CSV has only
a header. final.pt contains the model, Adam state, and logical step count.

Run tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n curve python -m pytest \
  -c exp1a/pytest.ini exp1a/tests --basetemp=exp1a/results/test_tmp -q
```

Smoke results are isolated in `exp1a/results/smoke/`: each method executes one
logical batch (20 x 50), evaluates 100 test examples, and is marked smoke=true.
They are never included in the formal sweep summary.

`run_sweep.sh` uses the active Python environment. The Python launcher queues
all 27 trials across GPU IDs 0,1,2,3, one process per GPU, and returns nonzero
if any trial fails. Each process gets its own train.log. sweep_summary.csv
stays in grid order and reports pending/running/completed/failed status.
