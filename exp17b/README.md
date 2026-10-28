# Exp17b: partial local curvature and deterministic Gaussian quadrature

Independent copy of Exp17's offline CIFAR-100 / ViT-Tiny mapping, augmentation,
FP32 ghost clipping, independent Poisson Gaussian release and RDP accounting.
Historical experiment files are read-only. All new artifacts live here.

`optimizer.py` implements Smooth, LC-alpha and GH-alpha with identical m/v state.
GH uses pre-step A/K, nodes 0,+/-sqrt(3)*nu and weights 2/3,1/6,1/6;
its inverse correction is (1+alpha)f - alpha*Qf. No added gradient query,
random probe, backward, clipping or history-curvature state. alpha=0 takes the
exact Smooth branch. r=0 Smooth is ordinary Adam; LC scans require r>0.

Run from repository root with the existing curve interpreter:

```bash
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp17b.prepare
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp17b.start_background
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp17b.status
tail -f exp17b/results/pipeline.log
```

Preparation freezes code/assets/environment, validates math, mechanism, model,
privacy and decisions, then performs four two-step real GPU smoke runs. Formal
horizon smoke uses all 50k training examples but INTERNAL validation only.
One scheduler lock; physical GPU 2 or 3 only, one worker per GPU. Occupied GPUs
wait, with no preemption. Each worker sees one device as cuda:0.

A: CPU float64, four prescribed clean paths, ten independent noisy histories per
path/time/SNR, 100,000 independent current draws per history, paired across all
r/method/alpha. History and current-noise streams are independent. Paths are
constant g, alternating signs ending toward g, linear-plus-sinusoidal, and
nonzero every fifth historical step. Exact histories are saved. The target is
f(g) using that noisy frozen history. r=0 LC is omitted. Oracle alpha is offline
analysis only. Pairwise MC intervals condition on each history, not adaptive
training. t=1 has empty history; ten independent current-noise repetitions.

B: seven configs x three seeds x 50 steps, using full 225-step noise multiplier.
Rank each correction method by mean validation CE, then accuracy, then alpha.
Keep its best alpha even if Smooth wins. C: three configs x five NEW seeds x 225.
C2: r=0/1 x Smooth/GH x three common seeds x 225; retain B's GH alpha.
Choose an extra r only if positive mean Smooth-minus-GH CE improvement exceeds
main r on the SAME three seeds; max improvement wins, tie lower r. This avoids
confounding a 3-seed ablation with a 5-seed mean. D: freeze all configurations
before any official test access; 3 (or 5) configs x ten NEW seeds x 250 steps.
No tuning on test, no hidden warmup, clipping or retry policy changes.

Checkpoints every ten steps include model, optimizer, accountant, all CPU/CUDA,
NumPy/Python/augmentation, sampler and private noise RNG states and diagnostics.
Restart the background launcher to resume interrupted trials. Successful reuse
requires matching source/assets/config and artifact hashes. Failed trials keep
tracebacks and halt scheduling; they are not silently retried.

See `results/protocol_freeze.json`, stage plans, `selected_alphas.json`,
`final_freeze.json`, per-trial config/checkpoint/progress/log files, and
`results/report.md`. Report updates after each complete stage, marks pending
stages, includes raw accuracies and percentage-point paired differences with
Student-t 95% intervals, and does not rank unfinished stages. Diagnostic and
curve CSVs retain update scale, direction, parallel/orthogonal correction,
privacy, runtime and memory. No claim of utility follows merely from MC bias.
Each training's epsilon is separate, not a joint epsilon=8 guarantee for tuning
and repetition. Validation is assumed public; raw audit artifacts are private.

Acceptance also simulates exit after the final checkpoint and before summary
creation, then verifies identical restored metrics and unchanged checkpoint.
Exp17's metadata tuple/list comparison and completed-checkpoint epsilon handling
were corrected locally. Earlier preparation attempts, their source, test logs,
and smoke artifacts remain under `runtime/preparation_attempt_*`,
`runtime/pre_resume_fix`, and `runtime/pre_metadata_fix`; they are not reused
under the final source freeze.
