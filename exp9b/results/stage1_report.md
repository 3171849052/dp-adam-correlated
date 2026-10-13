# Exp9b Stage 1

Platform tests and 22 real GPU smokes passed; Stage 2 has not started.
CV: seven methods at T=225 and T=250, two logical steps each (250 × 4). NLP: seven methods plus one additional concurrency smoke, two 1000-example steps each, max_length=128.
Two steps exercise previous completed vhat. Matrix/noise/accounting tests cover complete T=225/250/310.
Smoke is a feasibility and numerical check; it does not establish five-epoch accuracy or stability.

| Task | Horizon | Method | GPU | Peak allocated GiB | Peak reserved GiB | Seconds/step |
|---|---:|---|---:|---:|---:|---:|
| cv | 225 | dp-adam-iid | 0 | 7.759 | 8.131 | 2.545 |
| cv | 225 | dp-adam-sgd-bandmf | 1 | 7.780 | 8.152 | 2.520 |
| cv | 225 | dp-adam-momentum-bandmf | 2 | 7.780 | 8.152 | 2.481 |
| cv | 225 | dp-adam-momentum-bias-bandmf | 0 | 7.780 | 8.152 | 2.506 |
| cv | 225 | dp-adam-sgd-bandmf-scale | 1 | 7.986 | 8.516 | 2.293 |
| cv | 225 | dp-adam-momentum-bandmf-scale | 2 | 7.986 | 8.516 | 2.399 |
| cv | 225 | dp-adam-momentum-bias-bandmf-scale | 0 | 7.986 | 8.516 | 2.390 |
| cv | 250 | dp-adam-iid | 1 | 7.759 | 8.131 | 2.576 |
| cv | 250 | dp-adam-sgd-bandmf | 2 | 7.780 | 8.152 | 2.547 |
| cv | 250 | dp-adam-momentum-bandmf | 0 | 7.780 | 8.152 | 2.539 |
| cv | 250 | dp-adam-momentum-bias-bandmf | 1 | 7.780 | 8.152 | 2.463 |
| cv | 250 | dp-adam-sgd-bandmf-scale | 2 | 7.986 | 8.516 | 2.339 |
| cv | 250 | dp-adam-momentum-bandmf-scale | 0 | 7.986 | 8.516 | 2.342 |
| cv | 250 | dp-adam-momentum-bias-bandmf-scale | 1 | 7.986 | 8.516 | 2.343 |
| nlp | 310 | dp-adam-iid | 0 | 3.393 | 3.676 | 0.414 |
| nlp | 310 | dp-adam-sgd-bandmf | 0 | 3.410 | 3.678 | 0.386 |
| nlp | 310 | dp-adam-momentum-bandmf | 1 | 3.410 | 3.678 | 0.460 |
| nlp | 310 | dp-adam-momentum-bias-bandmf | 2 | 3.410 | 3.678 | 0.449 |
| nlp | 310 | dp-adam-sgd-bandmf-scale | 0 | 3.750 | 3.920 | 0.424 |
| nlp | 310 | dp-adam-momentum-bandmf-scale | 1 | 3.750 | 3.920 | 0.446 |
| nlp | 310 | dp-adam-momentum-bias-bandmf-scale | 2 | 3.750 | 3.920 | 0.474 |
| nlp | 310 | dp-adam-momentum-bias-bandmf-scale | 0 | 3.750 | 3.920 | 0.414 |

Expected full trainings: 321 grid + 56 recheck + 140 formal + 126 fixed-hyperparameter privacy scan = 643.
Per-run epsilon is verified individually; internal validation is assumed public/non-protected. The complete model-selection pipeline is not claimed to be epsilon=8 DP.
Historical CV test-based selection and previously viewed NLP official validation prevent a new blind-test claim.

From the repository root:

```bash
conda run --no-capture-output -n curve python -B -m exp9b.stage2 --gpus 0 1 2 --cv-per-gpu 1 --nlp-per-gpu 2
```
