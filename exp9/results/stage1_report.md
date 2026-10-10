# Exp9 Stage 1

Platform tests and 22 real GPU smokes passed; Stage 2 has not started.
CV: seven methods at T=225 and T=250, two logical steps each (250 × 4). NLP: seven methods plus one additional concurrency smoke, two 1000-example steps each, max_length=128.
Two steps exercise previous completed vhat. Matrix/noise/accounting tests cover complete T=225/250/310.
Smoke is a feasibility and numerical check; it does not establish five-epoch accuracy or stability.

| Task | Horizon | Method | GPU | Peak allocated GiB | Peak reserved GiB | Seconds/step |
|---|---:|---|---:|---:|---:|---:|
| cv | 225 | dp-adam-iid | 0 | 7.759 | 8.135 | 2.628 |
| cv | 225 | dp-adam-sgd-bandinvmf | 1 | 7.780 | 8.174 | 2.597 |
| cv | 225 | dp-adam-momentum-bandinvmf | 2 | 7.780 | 8.174 | 2.495 |
| cv | 225 | dp-adam-momentum-bias-bandinvmf | 3 | 7.780 | 8.174 | 2.472 |
| cv | 225 | dp-adam-sgd-bandinvmf-scale | 0 | 7.986 | 8.535 | 2.330 |
| cv | 225 | dp-adam-momentum-bandinvmf-scale | 1 | 7.986 | 8.535 | 2.342 |
| cv | 225 | dp-adam-momentum-bias-bandinvmf-scale | 2 | 7.986 | 8.535 | 2.313 |
| cv | 250 | dp-adam-iid | 3 | 7.759 | 8.135 | 2.556 |
| cv | 250 | dp-adam-sgd-bandinvmf | 3 | 7.780 | 8.174 | 2.692 |
| cv | 250 | dp-adam-momentum-bandinvmf | 0 | 7.780 | 8.174 | 2.533 |
| cv | 250 | dp-adam-momentum-bias-bandinvmf | 1 | 7.780 | 8.174 | 2.587 |
| cv | 250 | dp-adam-sgd-bandinvmf-scale | 2 | 7.986 | 8.535 | 2.292 |
| cv | 250 | dp-adam-momentum-bandinvmf-scale | 3 | 7.986 | 8.535 | 2.383 |
| cv | 250 | dp-adam-momentum-bias-bandinvmf-scale | 0 | 7.986 | 8.535 | 2.425 |
| nlp | 310 | dp-adam-iid | 0 | 3.393 | 3.676 | 0.365 |
| nlp | 310 | dp-adam-sgd-bandinvmf | 1 | 3.409 | 3.678 | 0.389 |
| nlp | 310 | dp-adam-momentum-bandinvmf | 2 | 3.409 | 3.678 | 0.361 |
| nlp | 310 | dp-adam-momentum-bias-bandinvmf | 3 | 3.409 | 3.678 | 0.486 |
| nlp | 310 | dp-adam-sgd-bandinvmf-scale | 0 | 3.750 | 3.920 | 0.360 |
| nlp | 310 | dp-adam-momentum-bandinvmf-scale | 1 | 3.750 | 3.920 | 0.425 |
| nlp | 310 | dp-adam-momentum-bias-bandinvmf-scale | 2 | 3.750 | 3.920 | 0.380 |
| nlp | 310 | dp-adam-momentum-bias-bandinvmf-scale | 3 | 3.750 | 3.920 | 0.476 |

Expected full trainings: 321 grid + 56 recheck + 140 formal + 126 fixed-hyperparameter privacy scan = 643.
Per-run epsilon is verified individually; internal validation is assumed public/non-protected. The complete model-selection pipeline is not claimed to be epsilon=8 DP.
Historical CV test-based selection and previously viewed NLP official validation prevent a new blind-test claim.

From the repository root:

```bash
conda run --no-capture-output -n curve python -B -m exp9.stage2 --gpus 0 1 2 3 --cv-per-gpu 1 --nlp-per-gpu 2
```

Additional verification: 61 unit tests passed. CLI dependency gates passed with mocked training; the report generator passed with explicitly synthetic fixtures under exp9/runtime/. Independent CPU reloads of all 22 real checkpoints verified finite model and Adam states, step=2, optimizer parameters, exact matrix inverse, actual full and prefix GDP budgets, and innovation tensor counts. See [artifact audit](artifact_math_audit.json), [workflow validation](workflow_validation.json), [environment](environment.json), and [platform gate](platform_validation.json).

Every physical GPU reached CV concurrency 1 and NLP concurrency 2. Highest per-process allocated/reserved memory: CV 7.986/8.535 GiB; NLP 3.750/3.920 GiB. Highest sampled whole-device memory was 8.896 GiB; samples every ~5 seconds can miss instantaneous peaks. Prior successful smoke artifacts (22 trials from an earlier platform source revision) and two initial unit-test failure/resolution records are retained in validation_archive/ and failures/. No full training was launched.
