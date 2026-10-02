# exp1b learning-rate sweep

Independent copy of exp1a: pretrained ViT-Tiny, FP32 full fine-tuning on existing CIFAR-100 (`download=False`). Fixed seed 20261001, classifier initialization, permutation and augmentation RNG setup across all five trials.

Adam: 3e-4, 5e-4. DP-Adam: 5e-4, 3e-3, 5e-3; clipping 1, epsilon 8, delta 1e-5. Five epochs, logical batch 1000, physical batch 250, 4 microbatches per update. Participation k/b and total steps are derived (5/50/250). Privacy math and exact pre-clipping norm collection are copied unchanged from exp1a; no sampling or amplification.

`run_sweep.sh` queues one process per GPU on 0–3, with at most four concurrent trials, records stdout/stderr and all artifacts under results, and exits nonzero if any trial fails. No best-trial selection. `sweep_summary.csv` uses final-epoch loss/norm statistics and all five epoch accuracies. Adam does not compute or save norm diagnostics.

Smoke runs one logical update and evaluates 100 test examples while retaining full-trajectory calibration. Formal sweep is not started by tests or smoke verification.

Run from repository root:

```bash
conda run --no-capture-output -n curve bash exp1b/run_sweep.sh
```
