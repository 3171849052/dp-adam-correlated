# Exp11

Run from the repository root:

```bash
bash exp11/run.sh > exp11/results/run.log 2>&1
```

The entry point uses the existing Conda `curve` environment, validates the exact
MSE objective and training semantics, performs a three-logical-step GPU smoke
with physical batch 250, then immediately executes 30 initial and up to eight
refined five-epoch CV trials, on physical GPUs 0,1,2,3 with one worker per GPU,
as subsequently requested by the user. No further confirmation
is needed. Failed tests or workers stop execution and expose the error.

Both objective terms use IID normalization. J2 contains both Gaussian-square
variance and squared bias and is calibrated at training clip=24. Four-tap
covariance diagonals include startup truncation. The full non-Toeplitz Exp9
momentum workload and fixed-participation absolute-Gram sensitivity are retained.
Every matrix is a numerical local solution, with BISR, IID and available
neighboring lambda initializations. Matrix caches are independent per lambda.

All writes are in exp11; original modules are vendored to avoid their package
initializers redirecting caches to exp9/exp10. Raw CIFAR training data and local
model weights are read only. The training-only loader deliberately avoids
torchvision CIFAR100's integrity check, which can also open the official test
file. The split is verified against Exp10. Seed 20261101, augmentation hashes,
model/head initialization hashes, order hashes and independent noise seed+1 are
paired across trials. No NLP or official-test evaluation runs.

The reports rank by fifth-epoch internal accuracy, then loss and lambda. They
are refreshed after each complete real training and explicitly marked incomplete
until the planned search ends. Tests' Monte Carlo samples validate a mathematical
identity only and are never used as training results. Completed jobs can be reused
only if code, assets and output hashes match; an existing failed job stops the queue.
Per-run epsilon=8 assumes public internal validation, without a composed search
privacy guarantee. One training seed provides no uncertainty estimate.

The active lambda=0 training from the original serial schedule is preserved.
results/gpu_transition.json records its original code hashes and the precise
switch to four GPUs. Its measurements and original metadata are not relabeled.
