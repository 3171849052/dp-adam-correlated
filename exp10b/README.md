# Exp10b: Adam-aware MF lambda ablation

Run from the repository root with the `curve` conda environment. Physical GPU
0 is rejected. Workers expose exactly one of physical GPUs 1/2/3 and use
`cuda:0`. All code, tests, runtime caches, matrices, logs and results are in
`exp10b/`; historical experiment packages are never imported. Data and local
pretrained weights are read only, with downloads disabled.

```bash
conda run --no-capture-output -n curve python -B -m exp10b.stage0 --gpus 1 2 3
conda run --no-capture-output -n curve python -B -m exp10b.run --gpus 1 2 3 --plan
conda run --no-capture-output -n curve python -B -m exp10b.run --gpus 1 2 3
```

Stage 0 prepares immutable splits, tokens and 18 matrices (three horizons,
six lambdas), runs tests and thirteen three-step GPU smokes. The CV smokes
cover lambda zero/positive at T=225/250. NLP first measures one worker on
each GPU and checks available memory, then actually runs two simultaneous
workers on each GPU. No full training or official evaluation runs in Stage 0.
`--plan` validates and prints the complete protocol without launching workers.

Stage 1 runs both tasks, every lambda `{0, .01, .03, .1, .3, 1}`, and seeds
20261101–20261103: 36 trainings. Stage 2 repeats every lambda with seeds
20261104–20261110: 84 trainings. All runs finish five epochs. Stage 3 selects
the best positive lambda separately for CV/NLP using mean final-epoch internal
validation accuracy across all ten seeds. Exact ties choose the smaller lambda.
It also reports the global winner including zero, then freezes configuration,
code, every matrix and all selection evidence by SHA256 before official data
are prepared. Stage 4 compares zero with that frozen positive lambda using
ten common seeds 20261111–20261120: 40 trainings. Total: 160 complete trainings.
No learning-rate, clipping, Adam or block-size search is performed.

Fixed CV: CIFAR-100/ViT-Tiny, LR=.003, C=30, five epochs, logical batch=1000,
physical batch=250 with four microbatches, T=225 internal and T=250 formal.
Fixed NLP: SST-2/BERT-Tiny, LR=.005, C=1, physical/logical batch=1000, T=310.
Standard PyTorch Adam uses beta1=.9, beta2=.999, eps=1e-8, weight_decay=0,
with both moment bias corrections. Every lambda uses the same Standard Ghost
clipping, initialization, sample ordering, CV augmentation and NLP dropout
rules vendored from Exp9. The independent Gaussian generator uses seed+1;
tests verify common standard innovations for every lambda over full horizons.
First-innovation signatures and paired initialization/order/augmentation
signatures are checked in actual GPU artifacts.

`D=I+d1 L+d2 L²+d3 L³`. Zero loads Exp9's actual saved matrix for each horizon
byte for byte; it is a real trained baseline and never invokes an optimizer.
Positive lambdas optimize `J1/J1(D0)+lambda*J2/J2(D0)` with C=1 using Exp10's
two fixed Nelder–Mead starts. J1 uses Exp9's full cumulative bias-corrected
Momentum workload (its convention omits the global 1-beta1 factor).
J2 is the exact Gaussian-noise-only bias-corrected second-moment variance,
including all cross-time covariance terms. Training recalibrates innovations
with the chosen strategy's fixed-participation absolute Gram sensitivity and
actual C. Every run independently verifies its full epsilon=8, delta=1e-5
calibration and each epoch's prefix expenditure; no sampling amplification.

Matrix NPZ files save coefficients, C_MF, inverse D, covariance and marginal
variances. Matrix JSON/CSV diagnostics save normalized J1/J2, J2_diag,
J2_cross and R_corr. Trial artifacts include configuration, epoch validation
metrics, DP preconditioner diagnostics, checkpoint, order, elapsed time and
peak allocated/reserved GPU memory. Diagnostics use the first 64 coordinates
of every parameter tensor; they only postprocess DP Adam state. No raw private
training losses, norms or clipping fractions are exported. Monte Carlo J2
validation evidence is in `results/math_validation/`.

The FIFO scheduler admits one CV process or up to two NLP processes per GPU,
with no task mixture. A process lock prevents duplicate launchers. Verified
completed trials are reused without retraining; incomplete or failed directories
stop the run and preserve evidence. There is no automatic retry, batch fallback,
early elimination or configuration mutation. Resume with the same run command
after inspecting failures; any manually archived failures must remain inside
`exp10b/`. Source changes invalidate Stage 0 and source-bound completed runs.

After internal selection, `selection_report.md` and plots show ten-seed
accuracy/95% Student-t CI, J1–J2 and J2 decomposition. After formal completion,
`report.md` separately compares the two formal groups with paired differences
and 95% CI. `exp9_cv_reproduction.json` checks actual same-seed Exp9 Standard
Momentum-Bias MF artifacts: initialization, order/augmentation, matrix,
noise calibration/protocol, final model tensors and evaluation results.
Differences are recorded explicitly; historical scores cannot affect selection.
Run `python -B -m exp10b.report` or `python -B -m exp10b.audit` to rebuild/audit
completed results.

Privacy is per training only. Internal validation is public/non-protected as
in Exp9; there is no epsilon=8 claim for the whole repeated-training/selection
pipeline. Historical CV test and NLP official validation have already been
viewed and are not new blind tests.
