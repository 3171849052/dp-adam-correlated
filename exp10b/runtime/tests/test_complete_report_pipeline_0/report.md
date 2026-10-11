# Exp10b formal paired comparison

Completed 36 + 84 + 40 = 160 full five-epoch trainings. All positive winners were frozen after all 120 internal runs and before official evaluation.

| Task | Lambda | Accuracy mean | Sample SD | 95% t CI |
|---|---:|---:|---:|---|
| cv | 0 | 0.700000 | 0.000000 | [0.700000, 0.700000] |
| cv | 0.01 | 0.710000 | 0.000000 | [0.710000, 0.710000] |
| nlp | 0 | 0.700000 | 0.000000 | [0.700000, 0.700000] |
| nlp | 0.01 | 0.710000 | 0.000000 | [0.710000, 0.710000] |

Paired differences compare the frozen best positive lambda against lambda zero on ten common seeds.
- CV: difference=0.010000, 95% CI [0.010000, 0.010000].
- NLP: difference=0.010000, 95% CI [0.010000, 0.010000].

CV same-seed Exp9 formal reproduction: 0/10 exact; details and any differences are in exp9_cv_reproduction.json.

Accuracy, validation loss, elapsed time, allocated/reserved GPU memory, matrix objectives and preconditioner changes are saved in the accompanying JSON/CSV files. Covariance and marginal variances are saved in matrix NPZ files.

Privacy applies separately to each epsilon=8, delta=1e-5 training, using Exp9 fixed-participation GDP calibration without sampling amplification. Internal validation is public/non-protected. No epsilon=8 guarantee is claimed for the complete selection/repetition pipeline. Only public evaluation and DP optimizer-state postprocessing diagnostics are released.

Historical CV official test and NLP official validation have already been viewed. These are not new blind tests. Exp9 comparison scores do not affect lambda selection.
