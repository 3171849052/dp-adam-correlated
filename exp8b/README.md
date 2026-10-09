# Exp8b: DP-Adam on BERT-Tiny / SST-2

Stage 1 prepares assets, runs unit tests and seven real one-step GPU smokes, then stops. Stage 2 is a separately invoked, complete search → freeze → 70-trial final → report pipeline.

```bash
conda run --no-capture-output -n curve python -B -m exp8b.stage1 --gpu 0
conda run --no-capture-output -n curve python -B -m exp8b.stage2 --gpu 0
# Authorized multi-GPU execution:
conda run --no-capture-output -n curve python -B -m exp8b.stage2 --gpus 0 1 2 --per-gpu 2
```

Fixed protocol: prajjwal1/bert-tiny, ordinary binary CLS/pooler classification, randomly initialized head, full fine-tuning, pretrained dropout 0.1, max_length=128, physical=logical batch=1000, FP32, 5 epochs / 310 updates, Adam (.9,.999), eps=1e-8, weight_decay=0. Train 67,349 records are stratified once with split seed 20261008 into 62,000 DP train / 5,349 search validation; one fixed permutation repeats across epochs (k=5, spacing=62). Official SST-2 validation (872) is read/downloaded only by final trials.

All implementation, tokenized trial inputs, logs, checkpoints, tests and outputs are under exp8b. Raw data is under data/sst2; model/tokenizer files under cache/bert-tiny. Asset revisions and hashes are pinned; existing assets can be reused. Runtime model/tokenizer loading is local-only. Existing experiments are read-only. No GPU locks or microbatch fallback. The default sequential FIFO runs one trial process at a time on the selected physical GPU; --gpus 0 1 2 --per-gpu 2 runs six isolated processes, at most two per GPU. Seven method searches advance independently on this shared queue; worker CUDA_VISIBLE_DEVICES maps it to local cuda:0.

## Mathematics

Methods: IID; SGD BandInvMF; Momentum BandInvMF; Momentum-Bias BandInvMF; the three corresponding Scale geometries. All MF matrices are rebuilt at T=310 with 4 bands. SGD and Momentum use the exact Exp7b/Exp2 JAX banded inverse-square-root filter construction at the new horizon; Momentum-Bias uses the same two-start, full-matrix float64 optimization with spacing=62. No CIFAR hyperparameters or results are imported.

SGD workload is cumulative prefix. Momentum workload is cumulative uncorrected first moment (the global constant 1-beta1 is omitted). Momentum-Bias uses W[t,j] = sum(beta1**(u-j)/(1-beta1**u), u=j..t), one-indexed. It is non-Toeplitz. D is a four-band lower triangular noising filter, S=D^{-1}; optimize sensitivity(S)^2 * ||W D||_F^2 / T with d0 fixed at 1. Bias means first-moment bias correction workload, not second-moment noise-bias correction. Actual S calibrates each mechanism to epsilon=8, delta=1e-5, add/remove zero-out with no sampling amplification. The absolute Gram bound allows distinct bounded gradient directions at each participation.

At step t Scale freezes s=1/(sqrt(vhat_(t-1))+eps_scale), uses ||s*g_i|| for clipping, adds correlated Gaussian noise to the scaled clipped sum, inverse-scales and divides by 1000 before ordinary Adam. eps_scale is distinct from Adam eps. Two backwards share one forward graph (including dropout). Embedding norms aggregate repeated (sample, token) derivatives before squaring, apply coordinate scales after aggregation, and remove padding derivatives: O(B*L*H) memory, never B*vocab*H. Linear Scale norms use exact layerwise sample gradients; an arbitrary coordinate scale does not satisfy the ordinary Ghost norm identity.

Within a seed all methods share model/head initialization, fixed order, tokenizer and dropout schedule (seed+100000+step). DP innovations use a dedicated generator (seed+1), independent of dropout. Epoch evaluations do not affect training dropout seeds.

## Search and freeze

Each of all seven algorithms searches seed 20261001 independently. Standard budgets ≤12 complete trials, Scale ≤18 including failed candidates. Initial LR candidates: 3e-5, 1e-4, 3e-4, 1e-3. Standard C: 0.1, 1, 10, 30, 100. Scale eps_scale: .03, .1, .3, 1. Coarse LR then C, Scale eps, then one geometric refinement per coordinate. Each boundary coordinate expands outward at most once. The bounded schedule stops at its budget. Scale C is generated from real smoke scaled norms, then the converging anchor's median scaled norm and clipping fraction; K=C*eps_scale only organizes eps candidates and does not imply equivalence.

The sole winner objective is search validation accuracy after epoch 5. Canonical trial ID breaks ties. Loss/norm/clipping/noise only generate diagnostics/candidates. Nonfinite trials are recorded and excluded; no mechanism changes. OOM aborts the platform, preserving physical batch=1000. All-failed candidate sets abort before freeze. Completed trials are audited before reuse; inspect other worker failures manually.

Each method writes candidate stages, a full accuracy curve, range provenance, winner, budget count and stop reason under results/search. selected_configs.json and frozen_configs.json contain all seven actual search winners; frozen_manifest.json and its .sha256 bind assets, code and search evidence. Finals enforce frozen hyperparameters and hashes. Stage 2 rechecks Stage 1 hashes/artifacts and reruns tests; source changes require rerunning Stage 1 (and invalidate an existing freeze).

Final seeds 20261011..20261020 produce exactly 7×10 full five-epoch trials, official validation accuracy only. Reports: final_multiseed.csv, method_summary.json, paired_effects.csv/json, final_report.md. Report raw accuracy, mean, sample std, SE, t(df=9) CI, eight paired comparisons and wins/10, plus three workload × two geometry table. Final best is descriptive and never feeds back into search.

Trial directory results/{smoke,search,final}/trials/<canonical-id> contains config.json, matrices.npz, train_order.npy, metrics.csv, mechanism_metrics.csv, checkpoint.pt (model plus Adam state), summary.json and train.log. Audits check epochs, updates/noise counts, physical batch, epsilon, data pairing, matrices, finite parameters/Adam states, pretrained/config/tokenizer hashes and frozen configuration. Stage 1 artifacts: unit_tests.log, tests.xml, smoke_summary.csv/json, platform_validation.json, stage1_report.md. No search or finals are run by Stage 1.

Single trial (after Stage 1):

```bash
conda run --no-capture-output -n curve python -B -m exp8b.train --method dp-adam-sgd-bandinvmf --lr 1e-4 --C 10 --category search --gpu 0
```

## Completed run

Stage 1 passed 41 unit tests, seven canonical GPU smokes and six additional simultaneous Scale smokes. The authorized Stage 2 run used GPU 0/1/2 with two processes per GPU and completed 102 search trials plus all 70 final trials.

Results: [final report](results/final_report.md), [raw final CSV](results/final_multiseed.csv), [search curves](results/search/search_curves.png), [final figure](results/final_summary.png), [completion audit](results/completion.json), [compute timings](results/compute_summary.csv).

The first parallel final launch exposed a shared temporary-download-file race. Four pre-training failures are archived under `results/failures/validation_download_race/`. The final runner now prepares official validation serially after frozen verification. The source-only manifest amendment preserves the original manifest and frozen hyperparameter hash; no LR, C, eps_scale, training kernel or search selection changed. All failed starts were rerun successfully.
