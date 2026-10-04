# Exp3: joint-DP Hybrid Muon + Adam

All implementation, logs, specs, caches and results are under `exp3/`. There are no imports from Exp1/2. Activate `conda activate curve` and run commands at the repository root.

`data/` supplies CIFAR-100 with `download=False`. The timm ViT-Tiny backbone is loaded strictly from `cache/huggingface/hub/models--timm--vit_tiny_patch16_224.augreg_in21k_ft_in1k/snapshots/<local refs/main>/model.safetensors`; missing files fail immediately. No pretrained network request is made. Checkpoint, initial model and classifier SHA256 digests are recorded.

## Mechanism

A full trial uses 5 epochs, 50 logical batches per epoch, 250 optimizer steps. Physical/logical sizes are 250/1000 with accumulation 4. One fixed training permutation is reused for all epochs to realize `(k,b)=(5,50)`. Each logical batch applies one full-model transformed norm, global clipping, one full-model IID or MF noise release, inverse geometry, and one hybrid optimizer step. Only the 48 Transformer qkv/proj/mlp.0/mlp.2 matrix weights use Muon; all other trainable parameters use Adam, with disjoint and complete partition checks.

Muon uses EMA momentum .95, Nesterov, Frobenius normalization with `1e-7`, standard quintic NS5 coefficients `(3.4445,-4.7750,2.0315)`, and shape scaling `sqrt(max(1,rows/cols))`. The optimizer and diagnostic map both use **FP32**; the [upstream Muon implementation](https://raw.githubusercontent.com/KellerJordan/Muon/master/muon.py) uses BF16. Adam uses `.9/.999/1e-8/0 weight decay`. The actual Nesterov matrix entering normalization is saved as `pre_ns` in optimizer state. Geometry freezes the previous completed matrix at the start of each logical batch.

All MF methods use the same ordinary momentum `.9` workload, four-band inverse-square-root coefficients and strategy; geometry is absent from factorization. GDP calibration follows Exp1/2, without sampling amplification and with add/remove zero-out adjacency. `innovation_std_sum = clip * fixed_epoch_sensitivity(strategy,5,50) / target_mu(8,1e-5)`, so changing clip recalibrates noise. Innovations are convolved by `D=C^-1` before inverse scaling. Adam and Muon share this one mechanism. The saved matrices include unscaled `D`, strategy, `W`, and `M = sigma D`; the noise coefficient stream is common to MF methods, while sigma changes with clip.

Version A scales the component parallel to normalized previous `H` by `sqrt(lambda_parallel)`. Version B uses an FP64 thin SVD; `delta=rho*mean(active singular_values(Hbar)^2)`. The median log-gain of the active singular spectrum gives one shared normalization center for both sides. Structural nullspace receives a finite ridge gain but does not set the center. The existing `[-log(kappa)/2, log(kappa)/2]` cap puts all pairwise scales within `[1/kappa,kappa]`. Both scale methods use identity on the first step; `lambda_parallel=1` and `kappa=1` give exact identity. Geometry only affects Muon blocks.

Norm computation uses Opacus Fast Clipping with exact layer-at-a-time transformed per-example gradients; no full-model Jacobian or full-model per-example gradient tensor is materialized. The second backward sums gradients using the single global clipping coefficient. Geometry is linear, so the sum is transformed at release time.

## Single trial

```bash
CUDA_VISIBLE_DEVICES=1 python -m exp3.train \
  --method mf_muon_spectralscale --seed 20261001 \
  --muon-lr 0.01 --adam-lr 0.0005 --max-grad-norm 100 \
  --kappa 4 --rho 0.1 --result-dir exp3/results/search/spectral_example

CUDA_VISIBLE_DEVICES=1 python -m exp3.run_trial --spec exp3/specs/trial.example.json
```

The JSON spec uses the same snake_case argument names. `result_dir` is required and must resolve inside `exp3/`. `smoke` defaults to false. Invalid parameters/unknown fields fail. `diagnostic_interval` defaults to 25 and `diagnostic_probes` to 4. For a direct trial, expose exactly one of physical GPUs 1/2/3. Existing complete trials with an identical spec are reused; incomplete/conflicting trials fail and are never deleted or repaired.

Each trial writes `config.yaml`, `train.log`, `metrics.csv`, `summary.json`, `train_order.npy`, `final.pt`, `matrices.npz`, plus `diagnostics.csv/json`. Accuracy is recorded as a fraction, not a percentage. Nonprivate runs have `epsilon/delta/clip_fraction=null`, zero innovation std and zero noise steps. Smoke privacy reports the actual two-step prefix epsilon; calibration still uses the full 250-step trajectory.

## Batch interface for Codex 6 Luna

Supply a JSON array containing 1–3 explicit trial specs, following `exp3/specs/batch.example.json`. Choose the next batch from `batch_summary.json` and the trial summaries. No grid or selection policy is encoded.

```bash
python -m exp3.launch_batch --specs exp3/specs/batch.example.json \
  --result-dir exp3/results/search/batch_example
```

Each child uses the current Python interpreter, one of GPUs 1/2/3 and no DDP; stdout/stderr go to its own `train.log`. Slots refill as soon as children finish. Any failed trial makes the batch exit nonzero. Complete results are reused. Batch outputs include the actual child specs and `batch_summary.json`.

## Explicit final validation

Fill the five-method mapping in `exp3/specs/frozen.json` using the format in `exp3/specs/frozen.example.json`. The example contains illustrative parameters, **not search-selected or validated configurations**. Freeze the actual parameters before invoking:

```bash
python -m exp3.final_runner --frozen-config exp3/specs/frozen.json \
  --result-dir exp3/results/final
```

This explicitly launches the 15 full trials using seeds 20261011/20261012/20261013 and max concurrency three. It snapshots and hashes the frozen config and refuses changed bytes on reruns. After completion it verifies initial state, classifier, checkpoint, full training order and augmentation trace pairing, then writes `final_summary.json` with mean and sample standard deviation. Every trial has its own `final.pt`; the frozen configuration must not contain `seed`, `method`, `result_dir` or smoke overrides.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp3/tmp" \
  python -m pytest -c exp3/pytest.ini exp3/tests -q > exp3/results/unit_tests.log 2>&1
python -m exp3.launch_batch --specs exp3/specs/smoke_mf.json \
  --result-dir exp3/results/smoke_v2/batch_mf
python -m exp3.launch_batch --specs exp3/specs/smoke_baseline.json \
  --result-dir exp3/results/smoke_v2/batch_baseline
python -m exp3.verify_smoke
```

The actual-pretrained, local-CIFAR GPU smoke runs each of five methods for two logical steps (8 physical batches, 2000 training examples, 100 test examples), checking both identity on the first step and adaptive geometry on the second. All three diagnostic layers use two fixed probes at both steps. These smoke specs do not start search or final validation.

JVP diagnostics evaluate `Phi` at completed `H_t` using the **actual frozen `S_t` from `H_(t-1)`**. They record per-layer probe mean/std/CV and p10/p50/p90, CV across layer means per diagnostic step, and temporal CV of each layer's mean. Probes use dedicated generators seeded by layer-name hashes; diagnostic RNG never affects training or noise. Norm/noise estimates and training logs are experiment diagnostics on local data; only the jointly noised gradient is the analyzed privacy mechanism.

## Active-spectrum normalization and diagnostics (revision 2)

Version B uses an FP64 thin SVD. Active singular values exceed `eps64 * max(shape) * sigma_max`. With `delta = rho * mean(active_sigma**2)`, a **shared** center is `median(-0.25 * log(active_sigma**2 + delta))`. Both sides subtract that center before the existing half-log-kappa cap. Structural nullspace gets the finite ridge gain and the same cap, and does not influence the center. Adding structural null dimensions leaves active gains unchanged. The trial fingerprint includes this implementation revision, so old artifacts cannot be mistaken for corrected runs; prior results remain untouched.

`diagnostics.csv/json` now contains `phi_gain_*`, `update_gain_*`, `noise_weighted_update_gain_*` for mean/std/CV/p10/p50/p90. Update gain includes the actual `sqrt(max(1,rows/cols))` factor. Cross-layer and temporal metrics are `layer_update_gain_cv` and `temporal_update_gain_cv`. Noise weighting is `innovation_std_sum/1000`; nonprivate values are null. No old Phi-only cross-layer metric is used as an optimizer metric.

MF trials save all 250 completed `H_t` matrices and actual frozen geometry for **only the three diagnostic layers** in `muon_trajectory.pt`. Inverse factors fully determine the saved spectral geometry. `frozen_trajectory_muon_mf.json` and the trial summary contain final cumulative RMSE and mean-prefix RMSE (mean/std/CV/p10/p50/p90 across probes), per-layer and equal-weight sampled-layer aggregate. Dedicated CPU temporal probes use the same layer/seed/sequence across methods, actual saved D, sigma/batch, actual EMA Nesterov .95 recurrence, JVP at frozen H, shape factor, and Muon LR before parameter-update accumulation. This is a **frozen-state first-order diagnostic, not exact nonlinear training dynamics**. It never affects optimizer or privacy calibration.

Recompute offline, for example:

```bash
CUDA_VISIBLE_DEVICES=1 python -m exp3.frozen_trajectory \
  --trial-dir exp3/results/search/<trial> --device cuda:0 --probes 2
```

`update_statistics.csv/json` records actual Muon/Adam released-gradient norms, update norms/RMS and relative update norms on diagnostic steps. Values also appear in `summary.json`. The existing two-step smoke specs now point to `exp3/results/smoke_v2/`, preserving the prior smoke. Its verification report is `exp3/results/smoke_v2_verification.json`.

`exp3/search_records.py` only records explicitly chosen sequential batches and their rationales. It enforces tuning seed 20261001, full five-epoch trials, stage ordering, budgets, one active batch, and result paths inside `exp3/results/search/`. It contains no search grid or automatic parameter policy. Exact `lambda=1` / `kappa=1` controls can reuse a completed equivalent Standard trial with the equivalence recorded, avoiding duplicate training. Search logs are `search_history.json`, `search_summary.csv`, `search_rationale.json`; frozen selected parameters are `selected_configs.json`.

Once all five stages close, `python -m exp3.search_audit` verifies full-trial counts, seed isolation, finite checkpoints, pairing, common MF matrices/probes and frozen configurations. `python -m exp3.search_report` exports the selected trials, utility differences and matching-LR/C Standard comparisons to `exp3/results/search/search_report.json` and `.md`. Neither command launches trials or reads final-validation results. The environment manifest is `exp3/results/search/environment.json`.

Run final validation separately after search has frozen the configuration:

```bash
conda activate curve
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp3/tmp" \
  python -m exp3.final_runner --frozen-config exp3/results/selected_configs.json \
  --result-dir exp3/results/final
```
