# Exp13 smoke numerical results

K=16; 4 independently fitted seeds; 3 independent holdouts per fit. Float64, no model execution.

J is the coordinate-averaged Frobenius workload objective with endogenous vhat. J_real is the measured mean squared cumulative pure-noise Adam update (learning rate 1). They are different functionals. All rows below evaluate actual N² denominators, including the matrix fitted by the expected-v comparator.

Holdout results average evaluations within each fitted seed before cross-seed statistics; 95% CIs use the Student t distribution. Relative improvements are paired percentages.

| T | split | method | J mean ± std | J_real mean ± std | J improvement vs IID (%) | J_real improvement vs IID (%) |
|---|---|---|---:|---:|---:|---:|
| 10 | in_sample | iid | 301.131 ± 487 | 8.25047 ± 2.95 | 0.000 | 0.000 |
| 10 | in_sample | exp9_momentum_bias | 275.085 ± 462 | 5.19272 ± 1.82 | 21.171 | 36.960 |
| 10 | in_sample | expected_self_consistent | 216.746 ± 387 | 0.3282 ± 0.0985 | 53.610 | 95.859 |
| 10 | in_sample | actual_self_consistent | 159.729 ± 307 | 1.02147 ± 0.897 | 77.575 | 88.956 |
| 10 | holdout | iid | 472.64 ± 678 | 8.01971 ± 0.187 | 0.000 | 0.000 |
| 10 | holdout | exp9_momentum_bias | 451.561 ± 669 | 4.89005 ± 0.148 | 25.665 | 38.677 |
| 10 | holdout | expected_self_consistent | 363.116 ± 645 | 0.35049 ± 0.0605 | 70.633 | 95.574 |
| 10 | holdout | actual_self_consistent | 399.818 ± 651 | 0.837407 ± 0.523 | 63.981 | 89.308 |

Coefficients, noise calibration, inverse amplification, vhat distributions, Exp9-relative improvements and all CIs are in metrics.csv/statistics.json. overfit_gaps.json reports in-sample minus holdout improvement: positive values indicate a loss of improvement on fresh innovations. paired_differences.json compares actual-noise fitting with all three comparators.
