# Exp8b search

Selection: seed 20261001 epoch-5 search-validation Accuracy only.

| Method | LR | C | eps_scale | Accuracy | Trials / budget | Stop reason |
|---|---:|---:|---:|---:|---:|---|
| dp-adam-iid | 0.001 | 0.31622776601683794 | None | 0.80202 | 12/12 | trial_budget_reached |
| dp-adam-sgd-bandinvmf | 0.003 | 10.0 | None | 0.82950 | 12/12 | trial_budget_reached |
| dp-adam-momentum-bandinvmf | 0.003 | 10.0 | None | 0.79510 | 12/12 | trial_budget_reached |
| dp-adam-momentum-bias-bandinvmf | 0.003 | 10.0 | None | 0.82015 | 12/12 | trial_budget_reached |
| dp-adam-sgd-bandinvmf-scale | 0.003 | 236.80201530456546 | 0.03 | 0.83847 | 18/18 | trial_budget_reached |
| dp-adam-momentum-bandinvmf-scale | 0.003 | 163.07783497332343 | 0.01 | 0.80501 | 18/18 | trial_budget_reached |
| dp-adam-momentum-bias-bandinvmf-scale | 0.003 | 216.7785169551871 | 0.03 | 0.83081 | 18/18 | trial_budget_reached |
