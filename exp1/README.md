# exp1

CIFAR-100 full fine-tuning of ImageNet pretrained timm
`vit_tiny_patch16_224.augreg_in21k_ft_in1k`. Every backbone parameter, cls token,
position embedding, LayerNorm and the randomly initialized 100-class head is
trainable. The Opacus-compatible implementation maps every pretrained backbone
tensor exactly: convolution weights become an equivalent unfolded-patch Linear,
and cls/position tensors become Embeddings. Unit tests compare timm outputs and
verify every trainable parameter contributes to the clipping norm.

Images are 224×224. Training uses RandomResizedCrop and RandomHorizontalFlip;
testing uses deterministic bicubic resize and center crop. Normalization comes
from the actual pretrained configuration (mean/std 0.5 for this checkpoint).
Only repository `data/cifar-100-python` is read, with `download=False`. Downloaded
pretrained assets are cached in `exp1/cache/`. Missing dependencies, weights,
data or GPUs raise errors; there is no random-initialization fallback.

## Schedule and mechanism

The input config does not contain manually specified k, b or total_steps.
Runtime derives `k=epochs`, `b=dataset_size//logical_batch_size`, `T=k*b`, and
requires dataset divisibility. Defaults yield (5,50,250). All matrix construction,
noise horizons and whole-trajectory GDP calibration use the derived schedule.

Adam uses beta1=0.9, beta2=0.999, eps=1e-8, no weight decay and FP32 arithmetic.
Each logical batch has 1000 examples: 20 physical microbatches of 50. Each DP
microbatch is clipped separately. Each logical batch advances noise/BandInvMF
and Adam exactly once. A seeded initial permutation is reused across epochs.
Every trial uses the same seed, pretrained initialization and permutation;
`initialization_sha256` and `train_order.npy` permit comparison.

`dp_adam` uses IID noise. Momentum BandInvMF uses
`toeplitz.multiply(ones(T), beta1**arange(T), n=T)`; Scale BandInvMF uses the
prefix-sum workload. Both have four noising bands. Innovations are independent;
noise is their finite convolution with the BandInvMF noising coefficients.

Privacy remains the whole-trajectory fixed-epoch absolute-Gram sensitivity bound
from JAX Privacy, with add/remove-zero-out adjacency and no sampling amplification.
GDP solves for target mu at epsilon=8, delta=1e-5; innovation standard deviation
on the sum is `max_grad_norm*sensitivity/target_mu`. Epoch diagnostics compute
sensitivity of the completed prefix under the original participation schedule.
There is no Opacus per-step GDP accountant.

Scale freezes `s=1/(sqrt(v_previous/(1-beta2**previous_step))+eps_scale)` for the
entire logical batch. Exact `||s*g_i||` is computed by layerwise per-example
samplers; arbitrary coordinate scales require Fast Clipping rather than the
ordinary Linear Ghost identity. The clipped accumulated sum is scaled, receives
correlated noise in scaled space, is inverse-scaled, and is divided by logical
batch size before Adam. Scale, norms, noise and Adam states remain FP32.

## Sweep

From the repository root, launch the full sweep with:

```bash
conda run --no-capture-output -n curve bash exp1/run_sweep.sh
```

The launcher automatically queues 36 single-GPU trials on GPUs 0,1,2,3, at most
four at once, without DDP or external scheduling tools. An available GPU takes
the next pending trial. It finishes all jobs and returns nonzero if any job fails.

Adam, DP-Adam and BandInvMF-Momentum each sweep lr in {1e-5,3e-5,1e-4}, with
DP clipping bound 1. Scale sweeps those rates × eps_scale in {1e-3,1e-2,1e-1}
× clipping bound in {0.3,1,3}, giving 27 Scale trials. No best setting is selected
using test results.

Results are in `exp1/results/sweep/<method>/<trial>/`: resolved `config.yaml`,
combined stdout/stderr `train.log`, epoch `metrics.csv`, `summary.json`, final
model/optimizer `final.pt`, `matrices.npz` (coefficients, strategy, workload), and
`train_order.npy`. `exp1/results/sweep_summary.csv` contains one row per trial,
including hyperparameters, seed, each epoch's test accuracy, final accuracy,
loss/clipping fraction, target epsilon/delta/mu, sensitivity, innovation std,
wall time, status and exit code. Rerunning overwrites files at the same paths.
Accuracy and clipping fractions are in [0,1]. Train losses/clipping statistics
are raw diagnostics; GDP describes the noised optimization stream.

## Validation

```bash
conda run --no-capture-output -n curve python -m pytest -c exp1/pytest.ini exp1/tests --basetemp=exp1/results/pytest_tmp -q
conda run --no-capture-output -n curve bash exp1/run_all.sh --smoke
```

Smoke runs all four methods on separate GPUs, one logical step each, with the
complete pretrained model, 20×50 training examples and 100 test examples. It
retains full-schedule calibration and writes to `exp1/results/smoke/<method>/`.
It does not launch the 36-trial sweep. Logs are `results/unit_tests.log` and
`results/smoke_launcher.log`; validated smoke metadata is `results/smoke_check.json`.
