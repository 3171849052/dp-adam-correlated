# Exp9 Stage 1

Platform tests and 22 real GPU smokes passed; Stage 2 has not started.
CV: seven methods at T=225 and T=250, two logical steps each (250 × 4). NLP: seven methods plus one additional concurrency smoke, two 1000-example steps each, max_length=128.
Two steps exercise previous completed vhat. Matrix/noise/accounting tests cover complete T=225/250/310.
Smoke is a feasibility and numerical check; it does not establish five-epoch accuracy or stability.

| Task | Horizon | Method | GPU | Peak allocated GiB | Peak reserved GiB | Seconds/step |
|---|---:|---|---:|---:|---:|---:|
| cv | 225 | dp-adam-iid | 0 | 7.759 | 8.135 | 2.602 |
| cv | 225 | dp-adam-sgd-bandinvmf | 1 | 7.780 | 8.174 | 2.581 |
| cv | 225 | dp-adam-momentum-bandinvmf | 2 | 7.780 | 8.174 | 2.522 |
| cv | 225 | dp-adam-momentum-bias-bandinvmf | 3 | 7.780 | 8.174 | 2.476 |
| cv | 225 | dp-adam-sgd-bandinvmf-scale | 0 | 7.986 | 8.535 | 2.319 |
| cv | 225 | dp-adam-momentum-bandinvmf-scale | 1 | 7.986 | 8.535 | 2.297 |
| cv | 225 | dp-adam-momentum-bias-bandinvmf-scale | 2 | 7.986 | 8.535 | 2.283 |
| cv | 250 | dp-adam-iid | 3 | 7.759 | 8.135 | 2.621 |
| cv | 250 | dp-adam-sgd-bandinvmf | 0 | 7.780 | 8.174 | 2.585 |
| cv | 250 | dp-adam-momentum-bandinvmf | 1 | 7.780 | 8.174 | 2.477 |
| cv | 250 | dp-adam-momentum-bias-bandinvmf | 2 | 7.780 | 8.174 | 2.458 |
| cv | 250 | dp-adam-sgd-bandinvmf-scale | 3 | 7.986 | 8.535 | 2.308 |
| cv | 250 | dp-adam-momentum-bandinvmf-scale | 0 | 7.986 | 8.535 | 2.303 |
| cv | 250 | dp-adam-momentum-bias-bandinvmf-scale | 1 | 7.986 | 8.535 | 2.421 |
| nlp | 310 | dp-adam-iid | 2 | 3.393 | 3.676 | 0.366 |
| nlp | 310 | dp-adam-sgd-bandinvmf | 2 | 3.409 | 3.678 | 0.359 |
| nlp | 310 | dp-adam-momentum-bandinvmf | 3 | 3.409 | 3.678 | 0.399 |
| nlp | 310 | dp-adam-momentum-bias-bandinvmf | 3 | 3.409 | 3.678 | 0.308 |
| nlp | 310 | dp-adam-sgd-bandinvmf-scale | 2 | 3.750 | 3.920 | 0.358 |
| nlp | 310 | dp-adam-momentum-bandinvmf-scale | 2 | 3.750 | 3.920 | 0.374 |
| nlp | 310 | dp-adam-momentum-bias-bandinvmf-scale | 3 | 3.750 | 3.920 | 0.368 |
| nlp | 310 | dp-adam-momentum-bias-bandinvmf-scale | 3 | 3.750 | 3.920 | 0.371 |

Expected full trainings: 321 grid + 56 recheck + 140 formal + 126 fixed-hyperparameter privacy scan = 643.
Per-run epsilon is verified individually; internal validation is assumed public/non-protected. The complete model-selection pipeline is not claimed to be epsilon=8 DP.
Historical CV test-based selection and previously viewed NLP official validation prevent a new blind-test claim.

From the repository root:

```bash
conda run --no-capture-output -n curve python -B -m exp9.stage2 --gpus 0 1 2 3 --cv-per-gpu 1 --nlp-per-gpu 2
```
