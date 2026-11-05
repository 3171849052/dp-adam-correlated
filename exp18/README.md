# Exp18

MNIST + small CNN trained from scratch, following the user's amended authorization. Dataset downloads and prepared tensors live in the repository-root data/ directory (excluded from Git); temporary files and experiment outputs stay in exp18/. Historical code is read-only. Reused Exp16a GDP and online B2LT noise; the matrix solver and adaptive-alpha optimizer follow the new stp-ex2.pdf. Unlike the PDF's RDP conversion, both IID and B2LT use the user-required GDP calibration. Momentum-Bias includes the PDF's 1-beta factor.

STP-Ex2 uses d0=1, global per-sample scaled clipping, a single noised query, scalar alpha with 24 bisections, and second moments of partially restored queries. Its parameter update uses bias-corrected preconditioned momentum without a second denominator. The older exp15d class named stpex2 is not this algorithm.

Fixed train/validation/test sizes 54000/6000/10000 throughout, including formal runs. The training permutation is seeded once per trial and repeated across three epochs to enforce participation interval 54. Test tensors are opened by training only after hyperparameter freeze. Preparation downloads/serializes test data but does not evaluate it. No augmentation, dropout, BatchNorm or pretrained weights. Adam epsilon is 1e-8; STP epsilon is searched. No weight decay, TF32, mixed precision or batch accumulation.

Pair selection averages IID and B2LT validation scores equally. Coarse: 170 pairs, seed 20261101. Fine: geometric coordinate neighbors of top two, at most 10 Adam + 10 Momentum + 14 STP pairs, seed 20261102. Boundary: up to two pairs/group, factor 0.5/2, lr/clip/eps priority, same fine seed. Review: two candidates/group, three new seeds 20261103-05; rank using only these three seeds. Formal: one shared configuration/group, seeds 20261111-15. At most 486 full trainings. Caps need not be exhausted.

Run from repository root using `/home/longt29/miniconda3/envs/curve/bin/python -B`:

- Acceptance: `-m exp18.stage1`
- Dataset preparation: `-m exp18.assets` (downloads to `data/MNIST/`, writes `data/train.pt`, `data/test.pt` and `data/split.npz`; asset manifest paths are relative to the repository root)
- Detached launch/resume: `-m exp18.start_background`
- Status: `exp18/results/status.json`
- Launcher PID/command: `exp18/results/background.json`
- Pipeline log: `exp18/results/pipeline.log`
- GPU/process telemetry: `exp18/results/gpu_telemetry.jsonl`
- Per-trial configs, curves, privacy, logs and checkpoints: `exp18/results/trials/<id>/`

The scheduler starts with two independent workers per physical GPU 1/2/3. After measuring trial memory, it may increase to four when utilization/free memory permit, or reduce capacity with memory pressure. CUDA_VISIBLE_DEVICES binds each worker; no DDP. OOM and other errors are recorded and stop dispatch; active workers finish. No batch/precision fallback.

Resume replays deterministic stage plans and reuses hash-verified complete trials. Incomplete/failed directories remain untouched and stop resume: inspect the error, confirm no worker is alive, then archive the directory under exp18/results/recovery before rerunning the launcher. This is trial-level recovery, not mid-trial continuation. Core code, protocol, PDF, asset manifest and matrix files are frozen with SHA256. The pipeline lock prevents concurrent schedulers.

Every training has epsilon=3, delta=1e-5; the entire search and repetitions do not jointly have epsilon=3. Validation is treated as public. Raw loss and clipping diagnostics are internal audit outputs, not additional DP releases. Report five-seed sample SD, Student-t 95% CIs and paired B2LT-minus-IID differences, without multiple-testing adjustment.

## 2026-11-05 并发修订

用户要求暂停后完成单卡压力测试。新方案覆盖上文的旧并发配置：物理 GPU 1/2/3 各12进程，总36；单卡测试上限16，17进程OOM。每进程预留768 MiB、每卡1536 MiB余量；按其他进程占用缩减，利用率低于90%再逐步恢复。详情见 `results/concurrency/report.md`。当前流水线保持停止。恢复沿用 `-m exp18.start_background`，151个完成任务经过校验后复用，中断现场已归档。
