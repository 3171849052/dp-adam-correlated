# Exp13 full numerical results

K=4096; 10 independently fitted seeds; 3 independent holdouts per fit. Float64, no model execution.

J is the coordinate-averaged Frobenius workload objective with endogenous vhat. J_real is the measured mean squared cumulative pure-noise Adam update (learning rate 1). They are different functionals. All rows below evaluate actual N² denominators, including the matrix fitted by the expected-v comparator.

Holdout results average evaluations within each fitted seed before cross-seed statistics; 95% CIs use the Student t distribution. Relative improvements are paired percentages.

| T | split | method | J mean ± std | J_real mean ± std | J improvement vs IID (%) | J_real improvement vs IID (%) |
|---|---|---|---:|---:|---:|---:|
| 225 | in_sample | iid | 51266.3 ± 8.77e+04 | 123.156 ± 2.56 | 0.000 | 0.000 |
| 225 | in_sample | exp9_momentum_bias | 59305.3 ± 1.07e+05 | 4.52119 ± 0.0825 | -2.723 | 96.329 |
| 225 | in_sample | expected_self_consistent | 73941 ± 1.44e+05 | 0.849561 ± 0.0142 | -11.262 | 99.310 |
| 225 | in_sample | actual_self_consistent | 51168.4 ± 8.77e+04 | 50.7884 ± 53.2 | 2.872 | 58.464 |
| 225 | holdout | iid | 50963.2 ± 6.28e+04 | 122.523 ± 1.79 | 0.000 | 0.000 |
| 225 | holdout | exp9_momentum_bias | 60352.5 ± 8.24e+04 | 4.49553 ± 0.0679 | -2.995 | 96.330 |
| 225 | holdout | expected_self_consistent | 79718.5 ± 1.27e+05 | 0.843516 ± 0.00877 | -12.415 | 99.311 |
| 225 | holdout | actual_self_consistent | 74825.2 ± 1.27e+05 | 51.0193 ± 53.5 | -6.901 | 58.456 |
| 250 | in_sample | iid | 51279.3 ± 8.77e+04 | 135.695 ± 2.79 | 0.000 | 0.000 |
| 250 | in_sample | exp9_momentum_bias | 59466.3 ± 1.07e+05 | 4.49658 ± 0.0798 | -2.629 | 96.686 |
| 250 | in_sample | expected_self_consistent | 73940.6 ± 1.44e+05 | 0.84723 ± 0.0143 | -11.047 | 99.375 |
| 250 | in_sample | actual_self_consistent | 51174.3 ± 8.77e+04 | 55.0899 ± 58.2 | 3.039 | 59.093 |
| 250 | holdout | iid | 50976.2 ± 6.28e+04 | 135.113 ± 1.98 | 0.000 | 0.000 |
| 250 | holdout | exp9_momentum_bias | 60547.5 ± 8.28e+04 | 4.47381 ± 0.0665 | -2.917 | 96.688 |
| 250 | holdout | expected_self_consistent | 79718.1 ± 1.27e+05 | 0.841396 ± 0.00881 | -12.211 | 99.377 |
| 250 | holdout | actual_self_consistent | 74850.4 ± 1.27e+05 | 55.4081 ± 58.5 | -6.798 | 59.085 |
| 310 | in_sample | iid | 51310.2 ± 8.77e+04 | 165.901 ± 3.19 | 0.000 | 0.000 |
| 310 | in_sample | exp9_momentum_bias | 59804.4 ± 1.08e+05 | 4.45218 ± 0.0779 | -2.379 | 97.316 |
| 310 | in_sample | expected_self_consistent | 73939.8 ± 1.44e+05 | 0.844925 ± 0.0145 | -10.545 | 99.491 |
| 310 | in_sample | actual_self_consistent | 51183.6 ± 8.77e+04 | 65.1491 ± 70.1 | 3.612 | 60.441 |
| 310 | holdout | iid | 51007.1 ± 6.28e+04 | 165.295 ± 2.46 | 0.000 | 0.000 |
| 310 | holdout | exp9_momentum_bias | 60958.2 ± 8.37e+04 | 4.43358 ± 0.0636 | -2.706 | 97.317 |
| 310 | holdout | expected_self_consistent | 79717.4 ± 1.27e+05 | 0.839365 ± 0.00895 | -11.739 | 99.492 |
| 310 | holdout | actual_self_consistent | 74908.3 ± 1.27e+05 | 65.5861 ± 70.5 | -6.469 | 60.435 |

Coefficients, noise calibration, inverse amplification, vhat distributions, Exp9-relative improvements and all CIs are in metrics.csv/statistics.json. overfit_gaps.json reports in-sample minus holdout improvement: positive values indicate a loss of improvement on fresh innovations. paired_differences.json compares actual-noise fitting with all three comparators.
