# Exp18 final report

MNIST 54000/6000/10000, CNN from scratch, 3 epochs, batch 1000, FP32.

Each training run: epsilon=3, delta=1e-5, GDP without sampling amplification. This is NOT a composed epsilon=3 guarantee for tuning, validation or repeated runs. Validation is treated as public; raw loss and clipping diagnostics are internal audit artifacts.

Five-seed sample SD and Student-t 95% intervals; paired differences share initialization, order and innovation RNG seeds. No multiple-comparison correction. Matrix error describes the same-query reference, not a counterfactual adaptive trajectory.

| Method | Mean | SD | 95% CI |
|---|---:|---:|---|
| adam-iid | 0.924080 | 0.007272 | [0.915051, 0.933109] |
| adam-b2lt-sgd | 0.933600 | 0.006394 | [0.925661, 0.941539] |
| momentum-iid | 0.882400 | 0.007815 | [0.872697, 0.892103] |
| momentum-b2lt-momentum | 0.882760 | 0.008224 | [0.872548, 0.892972] |
| stpex2-iid | 0.898920 | 0.006286 | [0.891115, 0.906725] |
| stpex2-b2lt-momentum-bias | 0.901820 | 0.007420 | [0.892607, 0.911033] |

Paired B2LT − IID:
- adam: 0.009520, 95% CI [-0.001552, 0.020592]
- momentum: 0.000360, 95% CI [-0.000996, 0.001716]
- stpex2: 0.002900, 95% CI [0.001003, 0.004797]
