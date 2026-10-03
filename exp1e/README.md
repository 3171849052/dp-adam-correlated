# exp1e

Independent CIFAR-100 full-finetuning experiment using locally cached pretrained
ViT-Tiny, FP32, 5 epochs, logical batch 1000, physical batch 250. Accumulation,
k, participation spacing and total steps are derived from configuration.
BandInvMF/Scale, transforms and fixed-epoch GDP accounting reuse the validated
implementations in exp1c/exp1d. Data download is disabled; pretrained loading is offline.

Stage 1 searches clips 100/150/200/300/500 at lr=1e-3, eps_scale=.1.
Stage 2 searches lr=5e-4/1e-3/2e-3/3e-3 at the selected clip, reusing the
stage-1 winner. Ties select the smaller parameter. Search seed is 20261001.
Stage 3 runs Adam, DP-Adam, Momentum BandInvMF and selected Scale BandInvMF
on seeds 20261011/20261012/20261013. Sample standard deviation uses ddof=1.

All results, checkpoints, diagnostics, logs and cache writes stay in exp1e.
One trial uses one GPU; four GPU queues (0–3) operate within each stage.
A failed trial makes the pipeline nonzero and prevents subsequent stages.
Completed trials are validated and reused on restart; incomplete trial directories
cause failure and require explicit user handling. Existing results are never overwritten.
Norm/coordinate arrays stay in RAM; only epoch aggregate diagnostics are persisted.

From the repository root:

```bash
conda run --no-capture-output -n curve bash exp1e/run_sweep.sh
```

Unit tests and short smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n curve python -m pytest -c exp1e/pytest.ini exp1e/tests --basetemp=exp1e/results/pytest_tmp
conda run --no-capture-output -n curve bash exp1e/run_sweep.sh --smoke
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n curve python -m exp1e.verify_smoke
```
