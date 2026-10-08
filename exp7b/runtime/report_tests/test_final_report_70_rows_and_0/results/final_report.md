# Exp7b final report

70 full runs, 10 shared seeds; hyperparameters frozen before final.
Accuracy and differences are fractions. CI uses Student t, df=9. Wins exclude ties.
The best method comparison is descriptive and selected by final mean; it does not change hyperparameters.

| Method | Mean | Sample std | SE | 95% CI |
|---|---:|---:|---:|---|
| dp-adam-iid | 0.60450 | 0.00303 | 0.00096 | [0.6023341494103319, 0.6066658505896682] |
| dp-adam-sgd-bandinvmf | 0.61495 | 0.00333 | 0.00105 | [0.6125675643513651, 0.6173324356486352] |
| dp-adam-momentum-bandinvmf | 0.62540 | 0.00363 | 0.00115 | [0.6228009792923982, 0.6279990207076017] |
| dp-adam-momentum-bias-bandinvmf | 0.63585 | 0.00394 | 0.00124 | [0.6330343942334313, 0.6386656057665685] |
| dp-adam-sgd-bandinvmf-scale | 0.64630 | 0.00424 | 0.00134 | [0.6432678091744647, 0.6493321908255355] |
| dp-adam-momentum-bandinvmf-scale | 0.65675 | 0.00454 | 0.00144 | [0.6535012241154977, 0.6599987758845022] |
| dp-adam-momentum-bias-bandinvmf-scale | 0.66720 | 0.00484 | 0.00153 | [0.6637346390565309, 0.670665360943469] |

## Raw Top-1 in seed order 20261011–20261020

- dp-adam-iid: [0.6, 0.601, 0.602, 0.603, 0.604, 0.605, 0.606, 0.607, 0.608, 0.609]
- dp-adam-sgd-bandinvmf: [0.61, 0.6111, 0.6122, 0.6133, 0.6144, 0.6154999999999999, 0.6166, 0.6177, 0.6188, 0.6199]
- dp-adam-momentum-bandinvmf: [0.62, 0.6212, 0.6224, 0.6236, 0.6248, 0.626, 0.6272, 0.6284, 0.6296, 0.6308]
- dp-adam-momentum-bias-bandinvmf: [0.63, 0.6313, 0.6326, 0.6339, 0.6352, 0.6365, 0.6378, 0.6391, 0.6404, 0.6417]
- dp-adam-sgd-bandinvmf-scale: [0.64, 0.6414, 0.6428, 0.6442, 0.6456000000000001, 0.647, 0.6484, 0.6498, 0.6512, 0.6526000000000001]
- dp-adam-momentum-bandinvmf-scale: [0.65, 0.6515, 0.653, 0.6545, 0.656, 0.6575, 0.659, 0.6605, 0.662, 0.6635]
- dp-adam-momentum-bias-bandinvmf-scale: [0.6599999999999999, 0.6616, 0.6631999999999999, 0.6648, 0.6663999999999999, 0.6679999999999999, 0.6696, 0.6711999999999999, 0.6728, 0.6743999999999999]

## Paired differences

| Comparison | Mean difference | Sample std | 95% CI | Wins / 10 |
|---|---:|---:|---|---:|
| dp-adam-sgd-bandinvmf - dp-adam-iid | 0.01045 | 0.00030 | [0.01023, 0.01067] | 10/10 |
| dp-adam-momentum-bandinvmf - dp-adam-sgd-bandinvmf | 0.01045 | 0.00030 | [0.01023, 0.01067] | 10/10 |
| dp-adam-momentum-bias-bandinvmf - dp-adam-momentum-bandinvmf | 0.01045 | 0.00030 | [0.01023, 0.01067] | 10/10 |
| dp-adam-sgd-bandinvmf-scale - dp-adam-sgd-bandinvmf | 0.03135 | 0.00091 | [0.03070, 0.03200] | 10/10 |
| dp-adam-momentum-bandinvmf-scale - dp-adam-momentum-bandinvmf | 0.03135 | 0.00091 | [0.03070, 0.03200] | 10/10 |
| dp-adam-momentum-bias-bandinvmf-scale - dp-adam-momentum-bias-bandinvmf | 0.03135 | 0.00091 | [0.03070, 0.03200] | 10/10 |
| dp-adam-momentum-bias-bandinvmf-scale - dp-adam-momentum-bandinvmf-scale | 0.01045 | 0.00030 | [0.01023, 0.01067] | 10/10 |
| best method - IID | 0.06270 | 0.00182 | [0.06140, 0.06400] | 10/10 |

## Three workloads × two geometries

Each cell uses its own frozen hyperparameters. Gains compare paired seeds.

| Workload | Standard mean | Scale mean | Paired scale gain | Gain 95% CI | Wins / 10 |
|---|---:|---:|---:|---|---:|
| SGD | 0.61495 | 0.64630 | 0.03135 | [0.03070, 0.03200] | 10/10 |
| Momentum | 0.62540 | 0.65675 | 0.03135 | [0.03070, 0.03200] | 10/10 |
| Momentum-Bias | 0.63585 | 0.66720 | 0.03135 | [0.03070, 0.03200] | 10/10 |

| Workload | Standard (lr, C) | Scale (lr, C, eps_scale) |
|---|---|---|
| SGD | (0.005, 100.0) | (0.005, 100.0, 0.1) |
| Momentum | (0.005, 100.0) | (0.005, 100.0, 0.1) |
| Momentum-Bias | (0.005, 100.0) | (0.005, 100.0, 0.1) |

| Workload | Four noising coefficients d | Sensitivity of S | Full-workload error |
|---|---|---:|---:|
| SGD | [1.0, -0.5, -0.125, -0.0625] | 2.848162888 | 13.352562500 |
| Momentum | [1.0, -0.95, -0.00125, -0.0011875] | 7.852751309 | 34.671727631 |
| Momentum-Bias | [1.0, -0.531619438, -0.077999812, -0.279553556] | 4.065608121 | 421.903162884 |

Standard clips/noises gradients; Scale uses previous-vhat scaled query coordinates then inverse-scales before Adam.
Within a workload, Standard and Scale share D and S=D^(-1). The Gaussian innovation standard deviation also depends on the frozen clipping C.
The error column is ||W D||_F^2 / 250 for each stated workload. Momentum-Bias uses the full non-Toeplitz bias-corrected first-moment workload.
All methods use add/remove zero-out adjacency, k=5, spacing=50, no sampling amplification, ε=8, δ=1e-5. Adam eps=1e-8 is separate from eps_scale.

## Frozen settings

```json
{
  "dp-adam-iid": {
    "method": "dp-adam-iid",
    "seed": 20261001,
    "lr": 0.005,
    "C": 100.0,
    "eps_scale": null,
    "source": "search_selected"
  },
  "dp-adam-sgd-bandinvmf": {
    "method": "dp-adam-sgd-bandinvmf",
    "seed": 20261001,
    "lr": 0.005,
    "C": 100.0,
    "eps_scale": null,
    "source": "search_selected"
  },
  "dp-adam-momentum-bandinvmf": {
    "method": "dp-adam-momentum-bandinvmf",
    "seed": 20261001,
    "lr": 0.005,
    "C": 100.0,
    "eps_scale": null,
    "source": "search_selected"
  },
  "dp-adam-momentum-bias-bandinvmf": {
    "method": "dp-adam-momentum-bias-bandinvmf",
    "seed": 20261001,
    "lr": 0.005,
    "C": 100.0,
    "eps_scale": null,
    "source": "search_selected"
  },
  "dp-adam-sgd-bandinvmf-scale": {
    "method": "dp-adam-sgd-bandinvmf-scale",
    "seed": 20261001,
    "lr": 0.005,
    "C": 100.0,
    "eps_scale": 0.1,
    "source": "search_selected"
  },
  "dp-adam-momentum-bandinvmf-scale": {
    "method": "dp-adam-momentum-bandinvmf-scale",
    "seed": 20261001,
    "lr": 0.005,
    "C": 100.0,
    "eps_scale": 0.1,
    "source": "search_selected"
  },
  "dp-adam-momentum-bias-bandinvmf-scale": {
    "method": "dp-adam-momentum-bias-bandinvmf-scale",
    "seed": 20261001,
    "lr": 0.005,
    "C": 100.0,
    "eps_scale": 0.1,
    "source": "search_selected"
  }
}
```
