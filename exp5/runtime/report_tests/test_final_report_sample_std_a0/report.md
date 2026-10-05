# Exp5 final three-seed comparison

| method | C | lr | mean top1 | sample std | best seed | worst seed |
|---|---:|---:|---:|---:|---:|---:|
| dp-sgdm-iid | 1.0 | 0.0005 | 0.200000 | 0.100000 | 20261013 | 20261011 |
| dp-sgdm-bandinvmf | 1.0 | 0.0005 | 0.200000 | 0.100000 | 20261013 | 20261011 |

| method | seed | final top1 | clip fraction | query norm | noise std | momentum norm |
|---|---:|---:|---:|---:|---:|---:|
| dp-sgdm-iid | 20261011 | 0.100000 | 0.100000 | 0.1 | 0.1 | 0.1 |
| dp-sgdm-iid | 20261012 | 0.200000 | 0.100000 | 0.1 | 0.1 | 0.1 |
| dp-sgdm-iid | 20261013 | 0.300000 | 0.100000 | 0.1 | 0.1 | 0.1 |
| dp-sgdm-bandinvmf | 20261011 | 0.100000 | 0.100000 | 0.1 | 0.1 | 0.1 |
| dp-sgdm-bandinvmf | 20261012 | 0.200000 | 0.100000 | 0.1 | 0.1 | 0.1 |
| dp-sgdm-bandinvmf | 20261013 | 0.300000 | 0.100000 | 0.1 | 0.1 | 0.1 |

Completed search trials: 26

Privacy: replace-one; epsilon=8; delta=1e-5; five sparse direct participations.
BandInvMF: momentum workload, beta=0.9, bandwidth=4. IID: identity temporal filter.

Full per-seed diagnostics and privacy/workload metadata: final_summary.json.
