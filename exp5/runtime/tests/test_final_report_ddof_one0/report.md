# Exp5 coordinate-normalized SGDM

| method | tau | C | lr | mean top1 | sample std (ddof=1) |
|---|---:|---:|---:|---:|---:|
| dp-coordnorm-sgdm-iid | 0.04 | 1 | 0.001 | 0.200000 | 0.100000 |
| dp-coordnorm-sgdm-bandinvmf | 0.04 | 1 | 0.001 | 0.200000 | 0.100000 |

| method | seed | final top1 | raw mean gradient norm | mean transformed norm | clip fraction | query norm | coherence | noise std | momentum norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dp-coordnorm-sgdm-iid | 20261011 | 0.100000 | 0.1 | 0.1 | 0.100000 | 0.1 | 0.1 | 0.1 | 0.1 |
| dp-coordnorm-sgdm-iid | 20261012 | 0.200000 | 0.1 | 0.1 | 0.100000 | 0.1 | 0.1 | 0.1 | 0.1 |
| dp-coordnorm-sgdm-iid | 20261013 | 0.300000 | 0.1 | 0.1 | 0.100000 | 0.1 | 0.1 | 0.1 | 0.1 |
| dp-coordnorm-sgdm-bandinvmf | 20261011 | 0.100000 | 0.1 | 0.1 | 0.100000 | 0.1 | 0.1 | 0.1 | 0.1 |
| dp-coordnorm-sgdm-bandinvmf | 20261012 | 0.200000 | 0.1 | 0.1 | 0.100000 | 0.1 | 0.1 | 0.1 | 0.1 |
| dp-coordnorm-sgdm-bandinvmf | 20261013 | 0.300000 | 0.1 | 0.1 | 0.100000 | 0.1 | 0.1 | 0.1 | 0.1 |

Trajectory diagnostic means over all 250 logical steps:

| metric | IID | BandInvMF |
|---|---:|---:|
| mean_raw_sample_norm | 0.1 | 0.1 |
| mean_transformed_sample_norm | 0.1 | 0.1 |
| transformed_norm_std | 0.1 | 0.1 |
| clip_fraction | 0.1 | 0.1 |
| mean_clip_factor | 0.1 | 0.1 |
| raw_mean_gradient_norm | 0.1 | 0.1 |
| query_norm | 0.1 | 0.1 |
| batch_coherence | 0.1 | 0.1 |
| noise_std | 0.1 | 0.1 |
| momentum_norm | 0.1 | 0.1 |
| update_norm | 0.1 | 0.1 |
| logical_step_seconds | 0.1 | 0.1 |

Calibrated target mu: 1.66603059785.
transformed_norm_std is the mean within-logical-batch sample std; epoch CSVs also retain pooled epoch statistics.
logical_step_seconds includes epoch loader startup; the Phase 1 report separately records steady-state timing.


BandInvMF minus IID mean top1: 0.000000

Replace-one step sensitivity=0.002; epsilon=8; delta=1e-5; five fixed sparse participations.
BandInvMF uses the exp2 momentum workload, beta=0.9, bandwidth=4.
Privacy calibration applies to a single run conditional on fixed configuration. The nonprivate scale probe, test-based selection, raw diagnostics, and combined releases are excluded.
Full seed diagnostics and accountant metadata: final_summary.json.
