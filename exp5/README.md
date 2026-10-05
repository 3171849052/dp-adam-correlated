# Exp5: DP-SGDM IID vs BandInvMF

Only `dp-sgdm-iid` and `dp-sgdm-bandinvmf` are implemented. Every example's
gradient is globally L2 clipped to C. Ten physical batches of 100 examples
produce one clipped sum; divide by 1000, draw temporal DP noise once, then
`m = 0.9*m + noisy_average; theta -= lr*m`. Weight decay is zero. No Adam,
coordinate normalization, update clipping, loss scaling, AMP or DDP.

The single clipping implementation is Exp2's verified ordinary two-backward
Ghost/Fast clipping (`exp2.scale.clipped_microbatch`), wrapped only to collect
diagnostics. The model is copied from Exp2 with runtime paths moved to Exp5.
All parameters train. Data is read from `data/` with `download=False`, weights
are loaded explicitly from the local `cache/` safetensors snapshot. Missing
assets fail directly; HF/Transformers are offline. Runtime writes stay in
`exp5/runtime/`; all results/logs/configuration snapshots stay in `exp5/results/`.

## Privacy and workload

Five epochs, 50 logical steps per epoch, 250 total steps. Shuffle once per
seed, reuse the exact logical-batch order every epoch. A record directly
participates at positions `j, j+50, j+100, j+150, j+200`. There is no sampling
amplification. Replace-one per-query sensitivity is `2*C/1000`.

For GDP target mu solving epsilon=8, delta=1e-5, IID average-space std is
`2*C*sqrt(5)/(1000*mu)`. BandInvMF reuses Exp2's sparse participation accountant
and `momentum_bandinvmf` Toeplitz coefficient construction, with bandwidth=4.
Workload coefficients are `sum(beta**i for i in range(t+1))`: prefix of the
unnormalized momentum recurrence. The full strategy's sparse sensitivity
multiplies `2*C/1000` before calibration. Noise history contains innovations,
never raw gradients. Momentum receives only the noisy logical average.

The privacy boundary is average plus noise; momentum is post-processing.
Train metrics and clipping/norm diagnostics are internal research outputs;
they are not separately privatized. Epsilon=8 is per training trial, not a
privacy claim for the entire tuning procedure. Selection uses final test top1.

## Tests and three-GPU smoke

From the repository root, in `curve`:

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp5/runtime/tmp" conda run --no-capture-output -n curve python -B -m pytest exp5/tests -q -o cache_dir=exp5/runtime/pytest_cache --basetemp=exp5/runtime/tests > exp5/results/unit_tests.log 2>&1
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp5/runtime/tmp" conda run --no-capture-output -n curve python -B -m exp5.launch_batch --smoke --gpus 0,1,2 > exp5/results/smoke_launcher.log 2>&1
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp5/runtime/tmp" conda run --no-capture-output -n curve python -B -m exp5.verify_smoke > exp5/results/smoke_verification.log 2>&1
```

Smoke runs three independent single-GPU trials simultaneously (IID, BandInvMF,
and a second IID seed), each one complete 1000-example logical step, followed
by evaluation on 100 test examples. Calibration always uses the full horizon.
Smoke results never enter search, freezing, or final reports.

Phase 1 validation completed in `curve`: 11 tests passed; all three real GPU
smoke runs passed `verify_smoke`. Paired IID/BandInvMF initialization and
pre-noise clipping/query diagnostics match. See `results/unit_tests.log` and
`results/smoke_verification.json`.

## Deterministic staged search

Only C and lr vary; every unique candidate trains for 5 epochs using seed
20261001. Stage 1: IID C in .25,.5,1,2,4 at lr=5e-4. Stage 2: best Stage 1 C,
lr in 1.25e-4,2.5e-4,5e-4,1e-3,2e-3. Stage 3: 3x3 grid of C and lr factors
.5,1,2 about the best IID candidate so far. Freeze IID. Stage 4: BandInvMF C
in .25,.5,1,2,4 at IID winner lr. Stage 5: best Stage 4 C and IID lr times
.25,.5,1,2,4. Stage 6: 3x3 grid about best BandInvMF candidate so far.
Freeze BandInvMF. Selection over all completed candidates of each method:
maximum final top1, then smaller C, then smaller lr. No early stop or pruning.

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp5/runtime/tmp" conda run --no-capture-output -n curve python -B -m exp5.search > exp5/results/search_launcher.log 2>&1
```

SHA256 IDs include method, seed, exact float lr/C, and the entire fixed
mechanism protocol. Completed trials are validated and reused, including
overlap across stages and reruns. Incomplete directories fail explicitly and
require inspection; no automatic repair or fallback. Every completed stage
immediately writes `results/search/stageN_summary.json`. Individual completed
trials update `search_summary.json` immediately. Frozen files are immutable:
`iid_frozen.json`, `bandinvmf_frozen.json`, `selected_configs.json`, all under
`results/search/`. Search finishes without starting final seeds.

## Six final trials and report

The batch entry reads frozen configurations and launches two methods x seeds
20261011,20261012,20261013. Physical GPUs 0,1,2 run at most one trial each.

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp5/runtime/tmp" conda run --no-capture-output -n curve python -B -m exp5.launch_batch --gpus 0,1,2 > exp5/results/final_launcher.log 2>&1
```

Each trial records train/test loss and top1, clipping fraction, mean raw
per-example gradient norm, mean clip factor, averaged clipped query norm,
noise std, momentum/update norms, spent epsilon/delta/mu, sensitivity, seed,
lr, C, method, strategy and workload metadata. `steps.csv` is per logical step;
`metrics.csv` is per epoch; `config.json`, `summary.json`, `matrices.npz`,
`train_order.npy`, `final.pt`, `train.log` allow auditing the run.

`results/final/final_summary.json` reports every seed's final top1 and
diagnostics, mean top1, sample std (ddof=1), best/worst seed, selected C/lr,
privacy/workload/bandwidth metadata, and completed search trial count.
`results/final/report.md` presents the comparison.
