# Exp6 final utility report

| Method | mean top1 | sample std | C | lr | geom_eps |
|---|---:|---:|---:|---:|---:|
| dp-lora-iid | 0.766933 | 0.002146 | 1.5 | 0.002 | inactive |
| dp-lora-bandinvmf | 0.812000 | 0.000781 | 5 | 0.015 | inactive |
| dp-lora-iid-scale | 0.769700 | 0.002138 | 0.3 | 0.002 | 1.0 |
| dp-lora-bandinvmf-scale | 0.814233 | 0.000896 | 5 | 0.02 | 1.0 |

| Seed | IID | MF | IID-scale | MF-scale |
|---|---:|---:|---:|---:|
| 20261011 | 0.765500 | 0.811500 | 0.767300 | 0.813200 |
| 20261012 | 0.769400 | 0.812900 | 0.771400 | 0.814700 |
| 20261013 | 0.765900 | 0.811600 | 0.770400 | 0.814800 |

Seed-paired effects (top1 fractions; sample std, ddof=1):

- MF_gain_raw: 0.045067 ± 0.001365; per seed [0.04600000000000004, 0.04349999999999998, 0.04569999999999996]
- MF_gain_scale: 0.044533 ± 0.001305; per seed [0.04590000000000005, 0.043300000000000005, 0.044399999999999995]
- scale_gain_iid: 0.002767 ± 0.001504; per seed [0.0018000000000000238, 0.0020000000000000018, 0.0044999999999999485]
- scale_gain_mf: 0.002233 ± 0.000839; per seed [0.0017000000000000348, 0.0018000000000000238, 0.0031999999999999806]
- interaction: -0.000533 ± 0.000666; per seed [-9.999999999998899e-05, -0.00019999999999997797, -0.0012999999999999678]

MF_gain_scale > MF_gain_raw: False (descriptive).

Full privacy, workload coefficients, clipping and LoRA geometry diagnostics are in report.json.
