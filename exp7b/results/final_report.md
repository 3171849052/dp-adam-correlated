# Exp7b final report

70 full runs, 10 shared seeds; hyperparameters frozen before final.
Accuracy and differences are fractions. CI uses Student t, df=9. Wins exclude ties.
The best method comparison is descriptive and selected by final mean; it does not change hyperparameters.

| Method | Mean | Sample std | SE | 95% CI |
|---|---:|---:|---:|---|
| dp-adam-iid | 0.59193 | 0.02423 | 0.00766 | [0.574593351662606, 0.609266648337394] |
| dp-adam-sgd-bandinvmf | 0.73276 | 0.00742 | 0.00235 | [0.727449072475646, 0.7380709275243541] |
| dp-adam-momentum-bandinvmf | 0.72585 | 0.01374 | 0.00435 | [0.7160182295984368, 0.7356817704015632] |
| dp-adam-momentum-bias-bandinvmf | 0.76423 | 0.00660 | 0.00209 | [0.7595099633182971, 0.7689500366817028] |
| dp-adam-sgd-bandinvmf-scale | 0.73480 | 0.00754 | 0.00239 | [0.7294026491316368, 0.7401973508683632] |
| dp-adam-momentum-bandinvmf-scale | 0.73534 | 0.01363 | 0.00431 | [0.7255890675378956, 0.7450909324621046] |
| dp-adam-momentum-bias-bandinvmf-scale | 0.76836 | 0.00663 | 0.00210 | [0.7636137568249681, 0.7731062431750317] |

## Raw Top-1 in seed order 20261011–20261020

- dp-adam-iid: [0.6198, 0.5494, 0.5895, 0.5721, 0.5798, 0.6246, 0.5991, 0.6028, 0.5697, 0.6125]
- dp-adam-sgd-bandinvmf: [0.7445, 0.7196, 0.7361, 0.7336, 0.7301, 0.739, 0.7307, 0.7368, 0.7226, 0.7346]
- dp-adam-momentum-bandinvmf: [0.7381, 0.6931, 0.7297, 0.7237, 0.7165, 0.7356, 0.7326, 0.7285, 0.7208, 0.7399]
- dp-adam-momentum-bias-bandinvmf: [0.7767, 0.7506, 0.7664, 0.7648, 0.7592, 0.7663, 0.7661, 0.7654, 0.7612, 0.7656]
- dp-adam-sgd-bandinvmf-scale: [0.747, 0.7216, 0.7376, 0.7355, 0.7323, 0.7403, 0.7327, 0.7405, 0.7245, 0.736]
- dp-adam-momentum-bandinvmf-scale: [0.7476, 0.7036, 0.7384, 0.7282, 0.7329, 0.7532, 0.7363, 0.7375, 0.7304, 0.7453]
- dp-adam-momentum-bias-bandinvmf-scale: [0.7794, 0.7544, 0.7698, 0.7655, 0.7641, 0.7728, 0.7699, 0.7729, 0.7656, 0.7692]

## Paired differences

| Comparison | Mean difference | Sample std | 95% CI | Wins / 10 |
|---|---:|---:|---|---:|
| dp-adam-sgd-bandinvmf - dp-adam-iid | 0.14083 | 0.01826 | [0.12777, 0.15389] | 10/10 |
| dp-adam-momentum-bandinvmf - dp-adam-sgd-bandinvmf | -0.00691 | 0.00885 | [-0.01324, -0.00058] | 2/10 |
| dp-adam-momentum-bias-bandinvmf - dp-adam-momentum-bandinvmf | 0.03838 | 0.00846 | [0.03233, 0.04443] | 10/10 |
| dp-adam-sgd-bandinvmf-scale - dp-adam-sgd-bandinvmf | 0.00204 | 0.00069 | [0.00155, 0.00253] | 10/10 |
| dp-adam-momentum-bandinvmf-scale - dp-adam-momentum-bandinvmf | 0.00949 | 0.00460 | [0.00620, 0.01278] | 10/10 |
| dp-adam-momentum-bias-bandinvmf-scale - dp-adam-momentum-bias-bandinvmf | 0.00413 | 0.00190 | [0.00277, 0.00549] | 10/10 |
| dp-adam-momentum-bias-bandinvmf-scale - dp-adam-momentum-bandinvmf-scale | 0.03302 | 0.00827 | [0.02710, 0.03894] | 10/10 |
| best method - IID | 0.17643 | 0.01856 | [0.16315, 0.18971] | 10/10 |

## Three workloads × two geometries

Each cell uses its own frozen hyperparameters. Gains compare paired seeds.

| Workload | Standard mean | Scale mean | Paired scale gain | Gain 95% CI | Wins / 10 |
|---|---:|---:|---:|---|---:|
| SGD | 0.73276 | 0.73480 | 0.00204 | [0.00155, 0.00253] | 10/10 |
| Momentum | 0.72585 | 0.73534 | 0.00949 | [0.00620, 0.01278] | 10/10 |
| Momentum-Bias | 0.76423 | 0.76836 | 0.00413 | [0.00277, 0.00549] | 10/10 |

| Workload | Standard (lr, C) | Scale (lr, C, eps_scale) |
|---|---|---|
| SGD | (0.002, 10.0) | (0.002, 100.0, 0.1) |
| Momentum | (0.005, 30.0) | (0.005, 100.0, 0.1) |
| Momentum-Bias | (0.003, 30.0) | (0.003, 100.0, 0.1) |

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
    "lr": 0.0005,
    "C": 30.0,
    "eps_scale": null,
    "source": "exp7_frozen",
    "num_bands": null,
    "provenance": {
      "selected_configs": "exp7/results/selected_configs.json",
      "selected_configs_sha256": "2ec62eda050805d310b6d1ae6031721ee64db02f4f6d44c5a45d03e7d6cc14f3",
      "original": {
        "seed": 20261001,
        "lr": 0.0005,
        "C": 30.0,
        "eps_scale": null,
        "final_test_top1": 0.5985,
        "source": "historical",
        "result_dir": "exp7/results/trials/93c00c98a74535cf",
        "num_bands": null,
        "utility": 0.5985,
        "stop_reason": "LR bracketed on both sides; historical C=10/30/100 bracket retained"
      }
    }
  },
  "dp-adam-sgd-bandinvmf": {
    "method": "dp-adam-sgd-bandinvmf",
    "seed": 20261001,
    "lr": 0.002,
    "C": 10.0,
    "eps_scale": null,
    "source": "search_selected",
    "num_bands": 4,
    "final_test_top1": 0.7391,
    "stop_reason": "LR .001/.002/.003 at C=10; stop at internal winner or after one directional extension",
    "provenance": {
      "result_dir": "exp7b/results/search/trials/cf989d0827b3b8be",
      "result_source": "historical",
      "historical_result_dir": "exp2/results/matched_search/prefix_standard_matched/clip_10",
      "objective": "seed=20261001 epoch-5 final_test_top1",
      "workload_sha256": "76ee4ed90f092c05441acf34dba914494413358bcccf4061549548609efaa956",
      "strategy_sha256": "d3d36864d39607dd3aeba9550886eb491d2e5ae3d3dc7edef408aca1f09e2ad9"
    }
  },
  "dp-adam-momentum-bandinvmf": {
    "method": "dp-adam-momentum-bandinvmf",
    "seed": 20261001,
    "lr": 0.005,
    "C": 30.0,
    "eps_scale": null,
    "source": "exp7_frozen",
    "num_bands": 4,
    "provenance": {
      "selected_configs": "exp7/results/selected_configs.json",
      "selected_configs_sha256": "2ec62eda050805d310b6d1ae6031721ee64db02f4f6d44c5a45d03e7d6cc14f3",
      "original": {
        "seed": 20261001,
        "lr": 0.005,
        "C": 30.0,
        "eps_scale": null,
        "final_test_top1": 0.7361,
        "source": "historical",
        "result_dir": "exp7/results/trials/272bde0be8d1d257",
        "num_bands": 4,
        "utility": 0.7361,
        "stop_reason": "frozen: historical C=10/30/100 and LR=.003/.005/.007 bracket; audited compatible"
      }
    }
  },
  "dp-adam-momentum-bias-bandinvmf": {
    "method": "dp-adam-momentum-bias-bandinvmf",
    "seed": 20261001,
    "lr": 0.003,
    "C": 30.0,
    "eps_scale": null,
    "source": "search_selected",
    "num_bands": 4,
    "final_test_top1": 0.7664,
    "stop_reason": "C search at LR=.005, then LR search at best C; at most one endpoint extension per coordinate",
    "provenance": {
      "result_dir": "exp7b/results/search/trials/2f5e945edc6cb786",
      "result_source": "new",
      "historical_result_dir": null,
      "objective": "seed=20261001 epoch-5 final_test_top1",
      "workload_sha256": "5789d76ad33c5d83b8a7d12d70472504b7d48545a405aa6fdb1e72bce392b229",
      "strategy_sha256": "3b5d25b76f731d3d145b56487662e1afe290ad9ed4ef20101be00148d10c7653"
    }
  },
  "dp-adam-sgd-bandinvmf-scale": {
    "method": "dp-adam-sgd-bandinvmf-scale",
    "seed": 20261001,
    "lr": 0.002,
    "C": 100.0,
    "eps_scale": 0.1,
    "source": "exp7_frozen",
    "num_bands": 4,
    "provenance": {
      "selected_configs": "exp7/results/selected_configs.json",
      "selected_configs_sha256": "2ec62eda050805d310b6d1ae6031721ee64db02f4f6d44c5a45d03e7d6cc14f3",
      "original": {
        "seed": 20261001,
        "lr": 0.002,
        "C": 100.0,
        "eps_scale": 0.1,
        "final_test_top1": 0.7402,
        "source": "new",
        "result_dir": "exp7/results/trials/a3883f283715f047",
        "num_bands": 4,
        "utility": 0.7402,
        "stop_reason": "epsilon/K/LR all bracketed at selected point"
      }
    }
  },
  "dp-adam-momentum-bandinvmf-scale": {
    "method": "dp-adam-momentum-bandinvmf-scale",
    "seed": 20261001,
    "lr": 0.005,
    "C": 100.0,
    "eps_scale": 0.1,
    "source": "search_selected",
    "num_bands": 4,
    "final_test_top1": 0.745,
    "stop_reason": "eps_scale=.1 wins K=10; compatible C/LR bracket retained",
    "provenance": {
      "result_dir": "exp7b/results/search/trials/5a41efa6b9e35903",
      "result_source": "historical",
      "historical_result_dir": "exp2/results/search/momentum_scale/lr_0.005_clip_100",
      "objective": "seed=20261001 epoch-5 final_test_top1",
      "workload_sha256": "5cb54d09d175b3b7d801f8a8b0e679140b8ee67c2db7b700419c314788546d63",
      "strategy_sha256": "b3bc96cab8b25da040b5352982fd3dec8682b72d131215bf9fa00284595508c7"
    }
  },
  "dp-adam-momentum-bias-bandinvmf-scale": {
    "method": "dp-adam-momentum-bias-bandinvmf-scale",
    "seed": 20261001,
    "lr": 0.003,
    "C": 100.0,
    "eps_scale": 0.1,
    "source": "search_selected",
    "num_bands": 4,
    "final_test_top1": 0.7703,
    "stop_reason": "eps neighborhood and K=5/10/20; one bounded endpoint refinement; LR at best (eps,C)",
    "provenance": {
      "result_dir": "exp7b/results/search/trials/eea4ff037b22783f",
      "result_source": "new",
      "historical_result_dir": null,
      "objective": "seed=20261001 epoch-5 final_test_top1",
      "workload_sha256": "5789d76ad33c5d83b8a7d12d70472504b7d48545a405aa6fdb1e72bce392b229",
      "strategy_sha256": "3b5d25b76f731d3d145b56487662e1afe290ad9ed4ef20101be00148d10c7653"
    }
  }
}
```
