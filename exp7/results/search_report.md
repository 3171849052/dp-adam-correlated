# Exp7 search-seed report

Status: completed; stage: done.
Seed: 20261001. Objective: epoch-5 final_test_top1 only. No final multi-seed runs.
New search trials: 19; historical reused: 15; numerical failures: 4.
Smoke trials are isolated and excluded from selection.

| Method | LR | C | eps_scale | Top1 | Source | Stop reason |
|---|---:|---:|---:|---:|---|---|
| dp-adam-iid | 0.0005 | 30 | None | 0.5985 | historical | LR bracketed on both sides; historical C=10/30/100 bracket retained |
| dp-adam-momentum-bandinvmf | 0.005 | 30 | None | 0.7361 | historical | frozen: historical C=10/30/100 and LR=.003/.005/.007 bracket; audited compatible |
| dp-adam-sgd-bandinvmf-scale | 0.002 | 100 | 0.1 | 0.7402 | new | epsilon/K/LR all bracketed at selected point |

| Scale stage | Best candidate LR | C | eps_scale | Top1 |
|---|---:|---:|---:|---:|
| coarse | 0.002 | 200 | 0.1 | 0.7393 |
| scale_joint | 0.002 | 100 | 0.1 | 0.7402 |
| scale_lr | 0.002 | 100 | 0.1 | 0.7402 |
| scale_refinement | 0.002 | 200 | 0.1 | 0.7393 |

All candidates (including numerical failures) are in search_summary.csv/json.
Scale coarse coordinate is K=C*eps_scale; workload is SGD/prefix-sum, and scale uses previous completed vhat.
Search is bounded at 28 new trials; one directional LR extension and one final local refinement are allowed.
Historical evidence was audited against local checkpoint, initialization, training order, matrices, calibration and protocol; see history_audit.json.
Runtime uses unchanged Exp2 mathematical kernels; full-size smoke verifies three GPU slots before search.

Checkpoint limitation (exp7/results/trials/534114dec78aa500): Initial model snapshot only (step 0); failed training state was not saved by the first failure handler.
