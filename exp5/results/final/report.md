# Exp5 coordinate-normalized SGDM

| method | tau | C | lr | mean top1 | sample std (ddof=1) |
|---|---:|---:|---:|---:|---:|
| dp-coordnorm-sgdm-iid | 0.037637131 | 1 | 0.016 | 0.122933 | 0.010627 |
| dp-coordnorm-sgdm-bandinvmf | 0.037637131 | 1 | 0.128 | 0.470067 | 0.044502 |

| method | seed | final top1 | raw mean gradient norm | mean transformed norm | clip fraction | query norm | coherence | noise std | momentum norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dp-coordnorm-sgdm-iid | 20261011 | 0.134500 | 3.22374 | 603.644 | 1.000000 | 0.0528782 | 0.0528782 | 0.00268431 | 14.3604 |
| dp-coordnorm-sgdm-iid | 20261012 | 0.120700 | 3.19159 | 603.445 | 1.000000 | 0.0619762 | 0.0619762 | 0.00268431 | 14.3641 |
| dp-coordnorm-sgdm-iid | 20261013 | 0.113600 | 3.0114 | 574.388 | 1.000000 | 0.0604478 | 0.0604478 | 0.00268431 | 14.3618 |
| dp-coordnorm-sgdm-bandinvmf | 20261011 | 0.514800 | 4.68258 | 738.967 | 0.999048 | 0.0449762 | 0.0450004 | 0.0129883 | 22.3525 |
| dp-coordnorm-sgdm-bandinvmf | 20261012 | 0.425800 | 5.02602 | 752.049 | 0.999520 | 0.0513879 | 0.0513987 | 0.0129883 | 22.353 |
| dp-coordnorm-sgdm-bandinvmf | 20261013 | 0.469600 | 4.77125 | 742.763 | 0.998984 | 0.0487265 | 0.0487534 | 0.0129883 | 22.3527 |

Trajectory diagnostic means over all 250 logical steps:

| metric | IID | BandInvMF |
|---|---:|---:|
| mean_raw_sample_norm | 69.921297 | 107.27346 |
| mean_transformed_sample_norm | 593.82561 | 744.59304 |
| transformed_norm_std | 77.861628 | 139.42529 |
| clip_fraction | 1 | 0.999184 |
| mean_clip_factor | 0.0017539759 | 0.0029595423 |
| raw_mean_gradient_norm | 3.1422419 | 4.8266153 |
| query_norm | 0.058434051 | 0.048363541 |
| batch_coherence | 0.058434051 | 0.048384174 |
| noise_std | 0.002684306 | 0.012988337 |
| momentum_norm | 14.362129 | 22.35275 |
| update_norm | 0.22979406 | 2.861152 |
| logical_step_seconds | 2.3278808 | 2.3510337 |

Calibrated target mu: 1.66603059785.
transformed_norm_std is the mean within-logical-batch sample std; epoch CSVs also retain pooled epoch statistics.
logical_step_seconds includes epoch loader startup; the Phase 1 report separately records steady-state timing.


BandInvMF minus IID mean top1: 0.347133

Replace-one step sensitivity=0.002; epsilon=8; delta=1e-5; five fixed sparse participations.
BandInvMF uses the exp2 momentum workload, beta=0.9, bandwidth=4.
Privacy calibration applies to a single run conditional on fixed configuration. The nonprivate scale probe, test-based selection, raw diagnostics, and combined releases are excluded.
Full seed diagnostics and accountant metadata: final_summary.json.
