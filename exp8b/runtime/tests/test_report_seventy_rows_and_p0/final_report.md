# Exp8b final report

70 full trials: 5 epochs, 310 logical steps, physical batch=1000, epsilon=8, delta=1e-5.
Accuracy is evaluated on official SST-2 validation (872 examples). Search used only the held-out 5,349 train examples.
Intervals use Student t(df=9), sample std(ddof=1); accuracy is a fraction. Wins exclude ties.
The final-best comparison is descriptive, chosen by final mean, without multiple-comparison correction; it never changes frozen hyperparameters.

| Method | Mean | Sample std | SE | 95% CI |
|---|---:|---:|---:|---|
| dp-adam-iid | 0.70450 | 0.00303 | 0.00096 | [0.7023341494103318, 0.706665850589668] |
| dp-adam-sgd-bandinvmf | 0.71450 | 0.00303 | 0.00096 | [0.7123341494103318, 0.716665850589668] |
| dp-adam-momentum-bandinvmf | 0.72450 | 0.00303 | 0.00096 | [0.7223341494103318, 0.726665850589668] |
| dp-adam-momentum-bias-bandinvmf | 0.73450 | 0.00303 | 0.00096 | [0.7323341494103318, 0.736665850589668] |
| dp-adam-sgd-bandinvmf-scale | 0.74450 | 0.00303 | 0.00096 | [0.7423341494103318, 0.7466658505896681] |
| dp-adam-momentum-bandinvmf-scale | 0.75450 | 0.00303 | 0.00096 | [0.7523341494103318, 0.7566658505896681] |
| dp-adam-momentum-bias-bandinvmf-scale | 0.76450 | 0.00303 | 0.00096 | [0.7623341494103318, 0.7666658505896681] |

## Raw accuracy in seed order 20261011–20261020

- dp-adam-iid: [0.7, 0.701, 0.702, 0.703, 0.704, 0.705, 0.706, 0.707, 0.708, 0.709]
- dp-adam-sgd-bandinvmf: [0.71, 0.711, 0.712, 0.713, 0.714, 0.715, 0.716, 0.717, 0.718, 0.719]
- dp-adam-momentum-bandinvmf: [0.72, 0.721, 0.722, 0.723, 0.724, 0.725, 0.726, 0.727, 0.728, 0.729]
- dp-adam-momentum-bias-bandinvmf: [0.73, 0.731, 0.732, 0.733, 0.734, 0.735, 0.736, 0.737, 0.738, 0.739]
- dp-adam-sgd-bandinvmf-scale: [0.74, 0.741, 0.742, 0.743, 0.744, 0.745, 0.746, 0.747, 0.748, 0.749]
- dp-adam-momentum-bandinvmf-scale: [0.75, 0.751, 0.752, 0.753, 0.754, 0.755, 0.756, 0.757, 0.758, 0.759]
- dp-adam-momentum-bias-bandinvmf-scale: [0.76, 0.761, 0.762, 0.763, 0.764, 0.765, 0.766, 0.767, 0.768, 0.769]

## Paired effects

| Comparison | Mean | Sample std | 95% CI | Wins / 10 |
|---|---:|---:|---|---:|
| dp-adam-sgd-bandinvmf - dp-adam-iid | 0.01000 | 0.00000 | [0.01000, 0.01000] | 10/10 |
| dp-adam-momentum-bandinvmf - dp-adam-sgd-bandinvmf | 0.01000 | 0.00000 | [0.01000, 0.01000] | 10/10 |
| dp-adam-momentum-bias-bandinvmf - dp-adam-momentum-bandinvmf | 0.01000 | 0.00000 | [0.01000, 0.01000] | 10/10 |
| dp-adam-sgd-bandinvmf-scale - dp-adam-sgd-bandinvmf | 0.03000 | 0.00000 | [0.03000, 0.03000] | 10/10 |
| dp-adam-momentum-bandinvmf-scale - dp-adam-momentum-bandinvmf | 0.03000 | 0.00000 | [0.03000, 0.03000] | 10/10 |
| dp-adam-momentum-bias-bandinvmf-scale - dp-adam-momentum-bias-bandinvmf | 0.03000 | 0.00000 | [0.03000, 0.03000] | 10/10 |
| dp-adam-momentum-bias-bandinvmf-scale - dp-adam-momentum-bandinvmf-scale | 0.01000 | 0.00000 | [0.01000, 0.01000] | 10/10 |
| final best - IID | 0.06000 | 0.00000 | [0.06000, 0.06000] | 10/10 |

## Three workloads × two geometries

| Workload | Standard mean | Scale mean | Paired Scale − Standard |
|---|---:|---:|---:|
| SGD | 0.71450 | 0.74450 | 0.03000 |
| Momentum | 0.72450 | 0.75450 | 0.03000 |
| Momentum-Bias | 0.73450 | 0.76450 | 0.03000 |

## Frozen provenance

Config SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
Manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
All trials passed matrix, privacy, training pairing, finite checkpoint/Adam, pretrained, tokenizer, split, and frozen-hash audits.
