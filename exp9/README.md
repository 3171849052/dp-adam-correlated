# Exp9: fixed-grid CV / NLP paper platform

All generated files live under `exp9/`. Historical experiments, `data/` and
`cache/` are read-only; no downloads, GPU locks, or batch fallback. Run from the
repository root using conda `curve` and `python -B`.

```bash
conda run --no-capture-output -n curve python -B -m exp9.stage1 --gpus 0 1 2 3
# Only prints/validates the implemented CLI and fixed counts:
conda run --no-capture-output -n curve python -B -m exp9.stage2 --gpus 0 1 2 3 --cv-per-gpu 1 --nlp-per-gpu 2 --plan
# Explicit opt-in: all 643 full trainings (do not run as part of Stage 1):
conda run --no-capture-output -n curve python -B -m exp9.stage2 --gpus 0 1 2 3 --cv-per-gpu 1 --nlp-per-gpu 2
```

Stage 1 prepares immutable stratified splits/tokens, runs mathematical and
workflow tests, and executes 22 real two-step GPU smokes. CV runs all seven
algorithms at both T=225 and T=250; NLP runs all seven plus one independent
additional smoke to fill two slots on every GPU. No official evaluation data
are read in Stage 1. Peak memory and cold step timings are feasibility evidence,
not five-epoch convergence evidence. Full-horizon tiny-tensor tests verify
noise draws and accounting at T=225/250/310.

`kernel_provenance.json` records the original verified source hashes. The ViT
model comes from Exp7b's Exp2 implementation; BERT, exact padding/repeated-token
Embedding clipping, two-backward training, FIR noise and GDP come from Exp8b.
Kernel copies prevent old package import side effects from writing history.
The full non-Toeplitz Momentum-Bias optimization uses spacing=T/5. IID's
workload is identity; SGD is prefix; Momentum is cumulative uncorrected first
moment; Momentum-Bias is cumulative bias-corrected first moment (global
`1-beta1` omitted). Standard/Scale share matrices within a workload/horizon.

Protocol: 5 epochs, batch 1000, Adam (.9,.999), eps=1e-8, decay=0, delta=1e-5,
add/remove zero-out, no sampling amplification, four MF bands. Scale freezes
previous completed vhat, clips exact scaled per-example gradients, adds FIR
noise in scaled sum space, inverse-scales, then divides by 1000.

CV: local pretrained ViT-Tiny, 224 preprocessing, full fine-tuning, 250×4
accumulation; official train stratified 45000/5000 with seed 20261008. Search
T=225/k=5/spacing=45; formal full 50000 training T=250/spacing=50. Official test
10000 is formal-only. NLP: local BERT-Tiny, random binary head, dropout .1,
full fine-tuning, max_length=128, physical=logical=1000; exactly reproduce
Exp8b's 62000/5349 split with seed 20261008; T=310/spacing=62. Official 872
validation is formal-only. Initialization and epoch-repeated training order
are seed-paired. CV uses seeded loader augmentation; NLP resets dropout to
seed+100000+step. DP innovations use an independent seed+1 generator.

Fixed grid specifications are in `grid.py`; generation writes both full JSON
and CSV. Seven methods independently select final-epoch internal validation
accuracy winners: 177 CV + 144 NLP complete search runs (seed 20261101), 56
Top-2 reruns (20261102/3), immutable 14-winner SHA256 freeze, 140 formal runs
(20261111..20), then 126 epsilon=2/4/16 fixed-hyperparameter runs (20261111..13).
No adaptive grids or old winners. The epsilon scan keeps epsilon=8 selected
LR/C/eps_scale, recalibrates only DP noise, and is explicitly not independent
tuning at each epsilon. Ties use canonical trial IDs.

One global FIFO, rotating GPU preference. Every physical GPU 0..3 participates;
CV at most one process/GPU, NLP at most two; no CV/NLP coexistence on a GPU.
Worker sees its assigned physical GPU through CUDA_VISIBLE_DEVICES and uses
local cuda:0. Scheduler events and nvidia-smi utilization samples are retained.
Only audited, matching completed artifacts are reused. Numerical search
failures are retained and excluded from Top-2; fewer than two successes abort.
Recheck/formal/sweep failures abort dependent stages. OOM preserves batch and
stops new launches while existing workers finish. Failed/incomplete directories
require inspection; move them within `exp9/results/failures/` before retrying.
There is no automatic numerical retry or complex recovery framework. Run a
single launcher at a time. Source changes require a new Stage 1; a freeze is
immutable and binds code, assets, grids and all selection evidence.

Outputs: per-trial config, epoch and step metrics, clip fraction, exact
calibration, matrix/order/checkpoint hashes, elapsed time, GPU and peak memory;
grid/Top-2/frozen tables; raw 10-seed official results, sample std, t 95% CI,
all paired method differences; main and workload/geometry tables; common
3-seed epsilon curves; grid heatmaps, timings, GPU utilization; audit and report.
`python -B -m exp9.audit` audits existing evidence, and
`python -B -m exp9.report` rebuilds a completed paper report.

Privacy claims apply individually to each training, never automatically to
the full selection/repeated-training pipeline. Search validation is assumed
public/non-protected; private validation or releases would require separate
accounting/protection. Historical CV selected with official test, and NLP's
official validation has been viewed previously. Do not claim a new blind test.
