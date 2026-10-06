# Exp5: per-example coordinate-normalized DP SGDM

Only `dp-coordnorm-sgdm-iid` and `dp-coordnorm-sgdm-bandinvmf` are implemented.

For each sample, `g_i = grad(loss_i)` and `h_i = g_i/(abs(g_i)+tau)`.
One norm across every parameter tensor determines `q_i = h_i min(1,C/||h_i||)`.
Ten physical batches of 100 form one logical batch of 1000. Their transformed,
clipped sums are averaged, then one noise vector is added. Momentum is
`m = .9*m + qbar + noise`, followed by `theta -= lr*m`. There is no Adam state,
Ghost clipping, batch-gradient normalization, weight decay, or `(1-beta)` factor.

`torch.func.grad_and_value` with `vmap` materializes only one chunk of per-example
gradients. The chunk is transformed, globally clipped, reduced, and released.
`CHUNK_SIZE` controls memory and is mathematically independent of the physical
batch size. `exp5/tests/test_coordnorm.py` compares several chunk sizes against
explicit per-example autograd.

All parameters of the locally cached pretrained ViT-Tiny are trainable. The head
is initialized for 100 classes. CIFAR-100 is loaded from `data/` with
`download=False`; pretrained weights are opened explicitly in `cache/`. No remote
pretrained loading is invoked. New output, caches, temporary files, locks and
checkpoints live under `exp5/`.

Each epoch uses the same seed-defined order. Augmentation is keyed by seed,
epoch and sample index, so interrupted trials can resume reproducibly from the
last complete epoch. Each epoch checkpoint contains model, momentum, innovation
history and noise RNG state. Unfinished epochs restart from that boundary.
Completed trial identities include method, seed, tau, C, lr and the entire fixed
mechanism configuration and are automatically reused. The scheduler locks GPUs
0,1,2 and launches at most one trial per GPU, with no DDP. Trial failures raise
and stop the scheduler; there are no algorithm or clipping fallbacks.

## Privacy calibration

Replace-one sensitivity of each logical average is `2*C/1000 = .002` with C=1.
IID innovation standard deviation is `.002*sqrt(5)/mu`, with mu calibrated to
epsilon=8, delta=1e-5. BandInvMF reuses `exp2.privacy.fixed_epoch_sensitivity`
with five participations spaced 50 steps apart, and
`exp2.bandinvmf.build_matrices('momentum_bandinvmf',250,4,.9)`. Its workload is
the prefix composed with unnormalized momentum. It does not use Exp4 accounting.
Momentum receives only the noisy aggregate and is post-processing.

The guarantee is a single run's fixed-configuration calibration. The nonprivate
scale probe and test-based selection are explicitly excluded. Raw diagnostic
releases and a combined release of multiple runs have no claimed aggregate
(epsilon=8) guarantee.

## Phase 1

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n curve python -B -m pytest exp5/tests -q -o cache_dir=exp5/runtime/pytest_cache --basetemp=exp5/runtime/tests
CUDA_VISIBLE_DEVICES=0,1,2 conda run --no-capture-output -n curve python -B -m exp5.launch_batch --smoke --gpus 0,1,2
conda run --no-capture-output -n curve python -B -m exp5.verify_smoke
```

Smoke runs one complete logical step, ten physical batches, then evaluates 100
test examples. Privacy noise remains calibrated for the full 250-step protocol.
The audit records peak allocated/reserved GPU memory and logical step time,
including initial loader startup. The extra IID seed checks GPU 2 independently.

## Phase 2

The deterministic nonprivate scale probe uses the first three physical batches
of the search seed's fixed order. It streams nonzero absolute gradient values
into an 8192-bin log10 histogram spanning [-50,10]. Quantiles p10,p25,p50,p75,p90
use interpolation inside bins; bin width gives the reported approximation
resolution. Full per-example gradients are never saved.

C=1 is fixed throughout. Every candidate trains five full epochs, with no early
stopping. Search stages are:

1. IID: five probe taus, lr=.001.
2. IID: best stage-1 tau, lr in {.00025,.0005,.001,.002,.004,.008}.
3. IID: adjacent available quantile-grid taus and best lr times {.5,1,2}.
4. Freeze IID; BandInvMF uses that tau, IID lr times {.25,.5,1,2,4}.
5. BandInvMF: same tau, best lr times {.5,1,2}; freeze.

Existing trial identities are deduplicated. Grid endpoints use available
neighbors. Winners maximize final test top1, then minimize lr, then maximize tau.
Five stage JSON summaries and the required scale_probe.json, iid_frozen.json,
bandinvmf_frozen.json, selected_configs.json and search_summary.json live in
`exp5/results/search/`.

Search, freeze, then automatically launch the six final runs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n curve python -B -m exp5.search --gpus 0,1,2 --run-finals
```

To run search only, omit `--run-finals`. To run or resume all frozen finals:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n curve python -B -m exp5.final_runner --gpus 0,1,2
```

Final seeds are 20261011,20261012,20261013. `results/final/report.md` and
`final_summary.json` report each seed, mean and sample std (ddof=1), selected tau
and lr, C=1, raw/sample/transformed norms, clipping statistics, query norm,
coherence, noise standard deviation, momentum/update norms, and privacy metadata.
Per-step and epoch CSVs contain train/test metrics (test evaluated each epoch),
tau, C, epsilon/delta/mu, sensitivity and timing. Coherence is
`||sum q_i|| / sum ||q_i||`; raw mean gradient norm means `||mean g_i||`, while
mean raw sample norm means `mean ||g_i||`. Diagnostic averages in the final
report cover the full trajectory; step CSVs preserve evolution.

Read current progress without allocating a GPU: `python -B -m exp5.status`.

Audit completed real runs and confirm cache reuse: `conda run --no-capture-output -n curve python -B -m exp5.audit_results`.
