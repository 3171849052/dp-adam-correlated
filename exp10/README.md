# Exp10

Two independent experiments: `adam-aware-mf` (standard Adam, optimized MF only) and `blockadam-mf` (Exp9 Momentum-Bias MF, frozen update denominator only). No IID, legacy baseline, combination, or epsilon sweep is trained.

The verified Exp9 model, Standard Ghost clipping, logical accumulation, FIR Gaussian innovations, GDP calibration, data splits and FIFO scheduler are vendored with namespace-only changes, followed by narrowly scoped Exp10 additions. `kernel_provenance.json` records the original file hashes. This avoids importing `exp9/__init__.py`, which sets writable runtime paths inside Exp9. All writes stay within Exp10; raw assets are read only from `data/` and `cache/`, with offline model loading and `download=False`. Split indices must equal Exp9. No raw private training loss, norm or clipping-fraction diagnostics are exported.

The latest user GPU instruction supersedes the initial four-GPU request: physical GPU 0 is excluded. The CLI rejects it. On each permitted GPU the FIFO scheduler admits either one CV worker or two NLP workers, never a mixture.

From the repository root:

```bash
conda run --no-capture-output -n curve python -B -m exp10.stage1 --gpus 1 2 3
conda run --no-capture-output -n curve python -B -m exp10.stage2 --gpus 1 2 3 --plan
conda run --no-capture-output -n curve python -B -m exp10.stage2 --gpus 1 2 3
```

Stage 1 only runs tests and ten three-step GPU smokes (four CV covering T=225/250; six NLP covering T=310 and two workers per GPU). Stage 2 requires passed Stage 1 with identical code/assets, precommits 96 initial search configurations, executes six nonduplicated internal-validation refinements per cell, reviews Top-2 with three new seeds, freezes four winners plus matrix/source/evidence hashes, executes forty paired formal runs, and generates reports. Total: 120 + 24 + 40 = 184 complete five-epoch trainings. Safe reuse only accepts completed, audited jobs; failures stop the queue, with no automatic retry or parameter fallback.

Initial CV pairs: (0.001,10), (0.001,30), (0.003,10), (0.003,30), (0.005,30), (0.003,100). NLP: (0.001,10), (0.003,10), (0.003,20), (0.005,10), (0.005,20), (0.005,1). Both methods share each task's pairs. These are drawn from Exp9's historical MF search ranges and include the specified historical winners. Refinements use three nearby unused LR/C changes around each of the initial Top-2. Ranking uses final internal accuracy, loss, and canonical trial ID. The refinement decisions are persisted before the refinement workers launch.

`J1` is Exp9's exact full cumulative Momentum-Bias workload objective, retaining its convention that W1 omits (1-beta1). `J2` is the average variance of the Gaussian-noise-only bias-corrected second moment. Objective normalization uses C=1; physical training calibrates sigma from the exact chosen strategy's fixed-participation sensitivity and the actual C. Each horizon/lambda matrix is computed once and persisted. Lambda zero is tested, not trained. Block size one is independently compared to PyTorch Adam, not trained. Pure cumulative first-moment noise diagnostics multiply W1 by (1-beta1)=0.1. Adaptive denominator diagnostics are DP-state postprocessing on the first 64 coordinates of each parameter tensor.

Reports do not claim that the complete selection pipeline is epsilon=8 DP, nor that historically viewed official evaluation sets constitute a new blind test. Per-run privacy assumes public/non-protected internal validation; epsilon accounting follows Exp9's add/remove-zero-out adjacency and absolute Gram sensitivity bound without sampling amplification.

A report integration check found and corrected a duplicate metadata keyword before formal runs. `source_compatibility.json` records the exact original source hashes and permits only this report/audit revision; every scientific training source must remain byte-identical. Existing completed searches and smokes are preserved with their original source hashes, never relabeled or rerun. The ledger itself is included in all subsequent source and freeze hashes. Synthetic report fixtures are isolated under `runtime/report_check/` and are not experiment results.
