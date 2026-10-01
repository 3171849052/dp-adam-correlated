# exp1

Four single-GPU CIFAR-100 / ViT-Tiny runs. All training arithmetic, Adam
moments, scales, norms and noise tensors are FP32, without AMP or pretraining.
The model uses 4×4 patches, width 192, depth 12, 3 attention heads, MLP ratio
4 and 100 classes. Training uses only the existing `data/cifar-100-python`,
with `download=False`. Images are normalized; no stochastic augmentation is
applied. Adam uses a constant learning rate of 0.001 and no weight decay.
Defaults, including clipping bound 1 and independent `eps_scale=0.001`, are
in `config.yaml`.

Run from the repository root:

```bash
conda run --no-capture-output -n curve bash exp1/run_all.sh
```

`run_all.sh` binds the four methods to visible GPU 0/1/2/3 respectively.
Each process sees its assigned GPU as `cuda:0`. It waits for every process
and returns nonzero if any fails. Full results go to `results/<method>/`.
Existing files at that destination are overwritten on another run.

## Matrix mechanism and privacy

For `D=C^{-1}`, the noisy logical-batch sum is
`clipped_sum[t] + sigma * sum_l D[l] z[t-l]`, with IID unit Gaussian
innovations. Division by 1000 happens after noising. BandInvMF uses the four
coefficients produced by JAX Privacy's
[`banded_inverse_square_root_noising_coefs`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.matrix_factorization.toeplitz.banded_inverse_square_root_noising_coefs.html).
These are bands of the **noising** matrix; the strategy is its full inverse.
The finite impulse response history stores three previous innovations.
The momentum workload is exactly `toeplitz.multiply(ones(T), beta1**arange(T), n=T)`.
The Scale method uses the default prefix workload. Neither includes beta2,
preconditioning or learning rates in the workload. `dp_adam` uses `D=C=I`.
JAX only constructs coefficients on CPU; PyTorch performs training and noising.

A seeded data permutation is generated once and reused every epoch, so
each example participates at `j, j+50, ..., j+200`. There is no sampling
amplification. Adjacency is add/remove represented by zeroing one record's
contribution in the fixed public schedule. The sensitivity bound is
`max_j sqrt(sum(abs((C.T@C)[P_j,P_j])))`, where `P_j=j+50*arange(5)`.
It permits different bounded gradient directions at different participations
and is exact for these nonnegative strategies. This is the fixed-epoch
absolute-Gram bound in JAX Privacy's
[`fixed_epoch_sensitivity`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.matrix_factorization.sensitivity.fixed_epoch_sensitivity.html).
IID noise therefore has sensitivity `sqrt(5)`, not `sqrt(250)`.

GDP solves
`delta = Phi(-epsilon/mu+mu/2) - exp(epsilon)*Phi(-epsilon/mu-mu/2)`
for target mu at epsilon 8 and delta 1e-5. Set
`sigma = clipping_bound * sensitivity(C) / target_mu` on the **sum**.
Per-epoch GDP values use the completed prefix of the strategy, with the
original `(5,50)` participation schedule. The final prefix is exactly 250
steps and reaches the configured target. No independent per-step composition
or Opacus sampling accountant is used. Adam has no DP guarantee and its GDP
fields are empty/null.

## Scale-then-Privatize

At the start of each logical batch, freeze
`s = 1 / (sqrt(v_previous/(1-beta2**previous_step)) + eps_scale)`;
the initial second moment is zero. The entire batch shares these scales.
First backward computes exact `||s*g_i||` via Opacus layer samplers, and
second backward computes `sum_i min(1,C/||s*g_i||)*g_i` with detached
coefficients. For arbitrary coordinate scales, a sequence Linear layer's
usual Ghost norm identity cannot be used. The scaled sampler uses **Fast
Gradient Clipping**, materializing per-example gradients for one layer at
a time and discarding them after norm reduction. Ordinary DP paths use
Opacus Ghost Clipping for supported layers and Fast Clipping for LayerNorm,
as in Opacus' [two-backward implementation](https://opacus.ai/api/grad_sample_module_fast_gradient_clipping.html).
The clipped accumulated sum is scaled, noised with BandInvMF in scaled
space, inverse-scaled, divided by 1000, and supplied to ordinary Adam.
There are no microbatch noise draws or microbatch Adam/MF updates.

## Validation and artifacts

```bash
conda run -n curve python -m pytest exp1/tests -q
conda run --no-capture-output -n curve bash exp1/run_all.sh --smoke
```

Smoke runs the complete model on two logical batches (40 microbatches,
2000 training examples) and evaluates 100 real test examples. It retains
the full 250-step calibration and writes only to `results/smoke/<method>/`.
It does not claim to complete the five-epoch schedule.

Each result directory contains complete resolved `config.yaml`, `train.log`,
`metrics.csv`, `summary.json`, `matrices.npz`, the fixed `train_order.npy`,
and `final.pt` (model and optimizer). `test_top1` and `clip_fraction` are
fractions in [0,1]. `noise_std` is the marginal noise standard deviation on
the **mean** gradient at the last epoch step, in the space named by
`noise_std_space`; Scale's inverse-scale produces coordinate-dependent
gradient-space noise. `innovation_std_sum` is the calibrated IID innovation
std before convolution and division by batch size.
Losses and clipping fractions are raw experiment diagnostics, not privatized
data releases; the GDP accounting describes the noised optimization stream.

Dependencies are the installed `curve` versions of PyTorch, torchvision,
Opacus, JAX, JAX Privacy, NumPy, SciPy, PyYAML and pytest. Missing data,
dependencies or GPUs raise errors. Unit test output is saved in
`results/unit_tests.log`; smoke validation is saved in `results/smoke_check.json`.
