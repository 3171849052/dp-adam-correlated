# Exp17 — CIFAR-100, independent Poisson DP-Adam comparison

Exact requested Adam, AdamBC, LC, HC formulas live in `optimizer.py`.
`cv_model.py`, `data.py`, `clipping.py` adapt the exp16a offline CV implementation;
historical packages are never imported. Local CIFAR-100, cached ViT-Tiny and the
exp9 fixed 45000/5000 split are read only. All runtime files stay under exp17.
No network download, NLP, MF, or temporally correlated noise is used.

From repository root, using conda curve's Python:

```bash
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp17.stage1
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp17.start_background
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp17.status
tail -f exp17/results/pipeline.log
```

The acceptance stage runs independent mathematical/mechanism/model checks,
fixed-path Monte Carlo measurements, and eight real two-step runs (four methods,
two horizons; only 64 internal validation examples). It freezes source hashes,
input hashes, Opacus calibrations and package versions. The full detached
pipeline runs 144 coarse + 48 fine + 16 new-seed review + 40 final trials.
Then 20 additional paired LC/HC smoothing-only ablations run separately.
Protocol and deterministic refinement rules are in `common.py` and frozen in
`results/protocol_freeze.json`. Each method has exactly 48 complete search runs.
Adam search configurations have three seeds; review adds two genuinely new
seeds. Other configurations have one search seed plus two review seeds.

Physical GPUs 1/2/3 each have at most one worker, isolated by
CUDA_VISIBLE_DEVICES. External GPU compute processes cause that GPU to wait.
Physical microbatch maximum is 250; actual sampled logical size varies and the
last microbatch may be shorter. Each step independently samples Bernoulli
memberships and normalizes once by fixed 1000 after adding one IID Gaussian
vector. Empty draws still perform noise/optimizer/accountant updates.
All parameters and optimizer tensors remain FP32; TF32/AMP are disabled.
Learning rate is constant, as in exp16a. The RDP accountant is solely Opacus.

Progress is in `results/pipeline_status.json` and per-trial `progress.json`.
`results/background.json` records the detached PID. A successful task is reused
only after matching code/input/config identities and checking every artifact
hash. Checkpoints atomically save model, optimizer (including HC R,H), accountant,
all RNGs, sampled indices, diagnostics and evaluations every ten logical steps.
To recover from interruption, rerun `start_background`; the last durable step is
restored and any later uncheckpointed work is replayed. Scheduling uses flock;
it will not start alongside an existing scheduler. An engineering/numerical
failure is preserved with traceback and stops scheduling new work, allowing
already-running workers to finish. It is never silently retried or modified.
Do not delete a failure or alter the frozen source to force reuse. Diagnose it
first; a changed implementation requires newly validated, separate evidence.

Reports (`report.md`, CSVs, training plot, final audit) are generated only after
complete verified final results. Ablations do not enter main method rankings.
Ten-seed confidence intervals use Student-t; differences are paired by seed,
without multiplicity correction. Curvature norm is relative to the smoothed
uncorrected update; cosine compares the actual direction to that baseline.
Monte Carlo reports measured biases, uncertainty, variance, MSE and log slopes;
it does not assert improvement. Small-noise order checks hold lambda2 positive
and fixed (varying r); lambda2=r*nu2 near zero has no uniform order guarantee.
LC conditions on preceding states; HC fixed-path reasoning does not prove
fourth-order unbiased adaptive training directions.

Epsilon 8 / delta 1e-5 applies to each individual training run, not the combined
search/review/repetition workflow. Internal validation is assumed public.
Raw gradient diagnostics and checkpoints are private audit artifacts, not a
DP-safe public release. Official test is read only after all four final configs
are frozen; prior historical use means it is not a newly blinded test set.

Preparation history: `results/preparation_audit.json` records the initial acceptance
and three interrupted preliminary search workers before a pipeline recovery fix.
Their source, logs and checkpoints are preserved under
`runtime/revisions/pre_ablation_resume_fix/`; none is reused by the current freeze.
