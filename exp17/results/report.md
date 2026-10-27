# Exp17: completed observations

Accuracy values are fractions; confidence intervals use Student-t across ten seeds. Paired intervals are unadjusted for multiplicity.

| Method | Accuracy mean | SD | 95% CI |
|---|---:|---:|---|
| adam | 0.725630 | 0.004446 | [0.722449, 0.728811] |
| bc | 0.722160 | 0.004382 | [0.719025, 0.725295] |
| lc | 0.730500 | 0.004602 | [0.727208, 0.733792] |
| hc | 0.724830 | 0.003688 | [0.722191, 0.727469] |

## Paired accuracy differences

| Comparison | Mean difference | 95% CI |
|---|---:|---|
| lc - adam | 0.004870 | [0.003827, 0.005913] |
| lc - bc | 0.008340 | [0.006972, 0.009708] |
| hc - adam | -0.000800 | [-0.002168, 0.000568] |
| hc - bc | 0.002670 | [0.001715, 0.003625] |
| hc - lc | -0.005670 | [-0.006699, -0.004641] |

## Audit and interpretation

{"coarse": 144, "fine": 48, "review": 16, "final": 40, "ablation": 20}
Opacus RDP per run only. Validation is assumed public; tuning, repeated runs, raw training diagnostics and checkpoints have no joint epsilon=8 release guarantee.
LC is conditional on fixed preceding states. HC fourth-order bias reasoning applies to fixed clipped-gradient paths with fixed positive smoothing, not adaptive training. Scaling lambda2 with nu2 is not a uniform fourth-order guarantee near zero.
Official test was opened only after all four final configs were frozen. Historical experiments have used this test set; this is not a newly blinded benchmark.
Detailed validation repetition counts: final_freeze.json. Per-step losses, corrections, cosine and norm diagnostics: training_curves.csv. Runtime, memory and cross entropy: runs.csv/utility.csv.
Calibration and environment: protocol_freeze.json. Monte Carlo bias/variance/MSE and measured slopes: monte_carlo.json.
Ablations: completed; see ablations.csv (20 additional trainings, excluded from main rankings).
