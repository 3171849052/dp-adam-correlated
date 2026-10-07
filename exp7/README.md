# Exp7

Search seed is exclusively 20261001; there is no final-seed runner. Model,
GDP calibration, BandInvMF matrices/noise and Algorithm-8 Scale come directly
from unchanged `exp2` modules. Runtime writes (including temporary files and
Python/pytest caches) stay in `exp7`. CIFAR-100 reads `data/`, and pretrained
weights read `cache/`, both offline.

From repository root:

```bash
conda run --no-capture-output -n curve python -B -m exp7.search --gpus 0 1 2
```

This runs tests, audits/imports historical points, runs/reuses three full-size
one-step GPU smokes, then automatically runs the complete staged search.
Exactly three distinct GPU slots form a FIFO; each slot has at most one trial.
Process and GPU locks prevent concurrent search launchers and overlapping trials on one GPU. Completed trial identities
are reused, while failed/incomplete directories require manual inspection and
cleanup. Numerical failures are recorded and excluded from accuracy selection.

```bash
conda run --no-capture-output -n curve python -B -m exp7.search --gpus 0 1 2 --smoke-only
conda run --no-capture-output -n curve python -B -m exp7.status
conda run --no-capture-output -n curve python -B -m exp7.report
conda run --no-capture-output -n curve python -B -m exp7.audit
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n curve python -B -m exp7.train --method dp-adam-sgd-bandinvmf-scale --lr .002 --max-grad-norm 200 --eps-scale .1
```

Every completed trial contains `config.yaml`, `metrics.csv`, `summary.json`,
`train_order.npy`, `matrices.npz`, `final.pt` and `train.log`, with Exp2 mechanism
traces as auxiliary diagnostics. Imported historical artifacts are copied to
Exp7 and retain original provenance in `historical_summary.json`.
`final.pt` names the epoch-5 checkpoint of a **search** trial; it does not mean
a final multi-seed experiment.

Search uses only epoch-5 Top1. IID retains the historical C bracket and adds
an LR bracket plus at most one directional extension. Momentum freezes its
verified historical bracket. Scale first searches epsilon at K=20/LR=.002,
then best epsilon and its neighbors at K=10/20/40, then LR=.0005/.001/.002/.003/.005.
One directional LR extension and one small final epsilon/K neighborhood are
allowed. The cap is 28 new search trials. Final artifacts record actual bracket
checks or bounded-refinement stop reasons; smoke is never eligible for selection.

Results: `exp7/results/selected_configs.json`, `search_summary.csv/json`,
`search_report.md`, `search_state.json`; scheduler transitions are in
`scheduler.jsonl`. Test and smoke evidence: `unit_tests.log`,
`history_audit.json`, `platform_validation.json` and `smoke/*/`.
