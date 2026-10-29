# exp16f

Six new STP methods, two tasks. Only exp16f is writable. All training uses the curve environment, offline data/ and cache/, FP32 and physical GPUs 1,2,3. CV uses one worker per GPU; NLP at most two, with no mixed tasks. Worker device is cuda:0.

The implementation copies the accepted exp16a STP logical-batch update and exp16d/e AdaM³ formula. Both scale using their own previous completed bias-corrected second moment, including zero on step one. AdaM³ accumulates the updated uncorrected momentum square. IID uses identical clipping/scaling/unscaling. Matrices are loaded read-only from exp16a with its acceptance hashes. Local tokenization and splits are regenerated from data/ and cache/ and checked against historical tensors. The historical exp16e formal comparison table is verified back to source seals, code, input assets and checkpoint hashes; its 20 Adam-STP baseline trials are reused without training.

Search: each task/method has 11 user-specified candidates. Boundary policy: inspect the best initial candidate; extend each active boundary once by multiplying its coordinate by 0.5 (lower) or 2 (upper), retaining other best coordinates. Deterministic priority lr, clip, eps_scale, at most two extensions per group, no recursive extension. Top two of the combined pool receive seeds 20261102/03. Select by three-seed mean accuracy, then mean cross entropy, then canonical ID. Freeze before opening official evaluation data for new training. Formal seeds 20261111–20261120.

Budget: 132 initial + 0–24 boundary + 48 review + 120 formal = 300–324 full training trials. Smoke tests are separate. Failures retain logs and stop; no parameter, batch or precision fallback. Completed trials are reused only after configuration, code, input, matrix and all artifact hashes pass. Partial runs require inspection; they are never silently reused or overwritten.

Run acceptance: `CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 /home/longt29/miniconda3/envs/curve/bin/python -B -m exp16f.stage1`

Launch/resume after acceptance: `bash exp16f/results/resume.sh`

Progress: `cat exp16f/results/progress.json`; `tail -f exp16f/results/pipeline.log`. Detailed worker PIDs, GPUs, commands and progress appear in `*_status.json`, `scheduler.jsonl`, and each trial's `diagnostics.jsonl`. `background.json` records the detached launcher. A lock prevents duplicate pipelines.

Single-run epsilon=8, delta=1e-5 does not imply that the search, validation, and repeated formal runs jointly satisfy epsilon=8. Theoretical errors describe scaled-space matrix noise, not the adaptive optimizer parameter error.

Recovery: use `bash exp16f/results/resume.sh`, which launches the hash-checked `exp16f.runtime.recover` entry point. It fixes selection of search records containing sealed numerical failures; failed records remain in the search evidence and cannot be ranked. Completed records still require explicit internal-evaluation metadata. The original training code, trial manifests, and logs remain unchanged. Recovery tests and verification evidence are in results/recovery_tests.log and results/recovery_acceptance.json.
