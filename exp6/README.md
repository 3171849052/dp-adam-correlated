# Exp6: DP-LoRA + Adam + temporal correlated noise

Stage 1 已完成平台与 smoke。Stage 2 使用 staged utility tuning，为四种方法分别选择 C/LR；两个 scale 方法共享 IID-scale 选择的 geom_eps。物理 GPU 仅使用 **1,2,3**，每卡最多一个训练进程。Search/freeze 和正式 12-trial launch 是两个显式独立动作。
## 固定协议

四种方法为 `dp-lora-iid`、`dp-lora-bandinvmf`、`dp-lora-iid-scale`、`dp-lora-bandinvmf-scale`。全部使用标准 `torch.optim.Adam`，betas=(0.9,0.999)、eps=1e-8、weight_decay=0。原 pretrained backbone 冻结；12 blocks 的 qkv/proj/mlp.0/mlp.2 共 48 个 Linear 注入 rank=8、alpha=8 的 LoRA，A Kaiming uniform、B=0。可训练参数为 314212 个，只有 96 个 A/B tensors 和 2 个分类头 tensors。

复用 `exp2.model` 的 `vit_tiny_patch16_224` 实现和严格 timm 权重映射，从 `cache/` 直接读取 safetensors。CIFAR-100 从 `data/` 读取，download=False。`runtime.py` 在导入训练依赖前设置 offline flags、临时文件和各类缓存位置；复用模型模块导入后恢复 Exp6 cache 环境。无需 peft，没有联网和下载路径。

训练为 5 epochs × 50 logical steps，logical batch=1000，Stage 1 默认 physical batch=8，Stage 2 plan 固定 physical batch=50；数据集大小 50000，总 250 steps。每个 seed 固定一个 permutation，每个 epoch 重用它，使同一样本在每个 epoch 参与同一 logical batch。增强由 seed/epoch/sample index 决定，不受 worker 调度、模型 RNG 和方法影响。输入预处理与 Exp2 相同。

每个 physical batch 用 `torch.func.vmap(grad_and_value)` materialize **可训练参数**的 per-example gradients，不为冻结 backbone 保存 per-example gradients。SDPA 使用 math backend，关闭 TF32，启用 deterministic algorithms。

Raw 在原梯度空间做 per-example global L2 clipping。Scale 在每个 logical step 开始时，detach 当前 A/B，FP64 eigendecomposition 计算

```text
P_A = sqrt(B^T B + geom_eps^2 I)
P_B = sqrt(A A^T + geom_eps^2 I)
g_A_scaled = P_A @ g_A
g_B_scaled = g_B @ P_B
```

冻结的 P 和逆矩阵转 FP32，在整个 logical batch 内保持不变。head geometry 为 identity。所有 LoRA scaled gradients 和 head gradients 共同做一次 per-example global clipping。聚合后在 query sum 空间加一次 temporal noise，然后用 P 的逆变换并除以 1000。Adam 的 m/v 直接从这些 unscaled noisy averages 更新，没有 vhat scale。`geom_eps` 是显式实验超参数；默认 0.1。

每个 logical step `TemporalNoise.draw()` 一次，向整个 trainable gradient vector 提供一次 Gaussian innovation；每个参数 tensor 的 randn 是这一个向量 draw 的分块。IID coefficients=[1]，strategy=noising matrix=I。BandInvMF 直接复用 `exp2.bandinvmf.build_matrices('momentum_bandinvmf',250,4,0.9)`；workload coefficients 为 prefix 和 beta1^t 的 convolution。Raw/scale 调用完全相同的构造，不根据 geometry 优化矩阵。同 seed 同 trainable tensor registration order 使用同 Gaussian seed，保存 standardized innovation SHA256 验证配对。

Privacy 直接复用 Exp2 GDP accountant、absolute-Gram fixed participation bound 和 add/remove zero-out adjacency。每个样本在 query 空间 clip 到 C，innovation std sum = C × fixed participation sensitivity / target mu；无 sampling amplification。完整 250-step 的四种机制统一 calibrate 到 epsilon=8, delta=1e-5。Smoke 使用完整机制的前三行，不重新校准为短实验，因此不同 temporal mechanisms 的实际 prefix epsilon 可以不同。逐步 epsilon/mu 按 full participation schedule 的 prefix 计算。

## 独立入口与结果

- `config.py`：固定协议、Trial identity、显式 lr/C/geom_eps/physical batch。
- `runtime.py`、`data.py`、`model.py`、`lora.py`：离线资源、确定性数据、冻结 backbone 与 LoRA 注入。
- `geometry.py`、`clipping.py`、`privacy.py`、`mechanism.py`、`train.py`：独立实验数学与训练入口。
- `launch_batch.py`：GPU 1/2/3 FIFO，每张 GPU 最多一个训练进程，使用 `runtime/gpu_*.lock` 防止两个 Exp6 launchers 重叠。
- `search.py`、`search_plan.json`：Stage 2 分阶段独立 C/LR 搜索与共享 scale geometry 的冻结选型。
- `final_runner.py`、`report.py`：冻结配置后多 seed factorial，paired effects、sample std (ddof=1)。
- `verify_smoke.py`、`tests/`：数学 invariants、pretrained 输出一致性、privacy/矩阵/metadata 审计。

每个 trial 保存 `config.json`、`train_order.npy`、`matrices.npz`、`steps.jsonl`、`geometry.jsonl`、`summary.json`、`final.pt` 和 launcher 的 `train.log`。配置含完整超参数、固定协议、seed、参数顺序、初始化 digest、checkpoint/data/source SHA256、软件版本、GPU、determinism、augmentation 约定和 privacy calibration。矩阵文件含 strategy、noising matrix、workload、所有 coefficients 及 innovation std。

`steps.jsonl` 记录 loss/top1、raw/scaled per-example norm 的 mean/min/max、clipping fraction、clip factor、query norm、noise std、Adam input/update norm、epsilon/delta/mu、A/B norms、actual-weight energy 和几何/whitening 诊断。`geometry.jsonl` 保留每层 P_A/P_B eigenvalues 的 extrema、condition、A/B norms 和 whitening Gram error。这些诊断使用私有训练信息，属于内部审计记录；privacy target 对应训练机制，不包括额外发布这些诊断。

`final.pt` 只保存 trainable tensors、Adam state、noise history/generator state、训练 RNG 状态；原 backbone 可由缓存 checkpoint 重建。当前入口支持完整重跑和复用已完成 trials，不提供中途恢复。不完整目录直接报错，由操作者显式处理，不静默覆盖或 fallback。

## 从仓库根目录执行

为让 conda 自身的 wrapper 临时文件也进入 Exp6，以下命令显式设置 TMPDIR。当前目录下的 `exp6/runtime/tmp` 已存在。

单元测试：

```bash
TMPDIR="$PWD/exp6/runtime/tmp" conda run -n curve python -B -m pytest -c exp6/pytest.ini exp6/tests -q
```

四方法、三 GPU smoke run 和 metadata 审计：

```bash
TMPDIR="$PWD/exp6/runtime/tmp" conda run -n curve python -B -m exp6.launch_batch --smoke --gpus 1,2,3
TMPDIR="$PWD/exp6/runtime/tmp" conda run -n curve python -B -m exp6.verify_smoke
```

已完成的 smoke trials 会被核对后复用。结果位于 `exp6/results/smoke/`，核验在 `exp6/results/smoke_verification.json`。本次 3-step smoke 的 IID prefix epsilon=3.1232925162，BandInvMF prefix epsilon=1.3407983831；两者完整 250-step calibration 均为 8。

Stage 2 搜索计划检查，不训练：

```bash
TMPDIR="$PWD/exp6/runtime/tmp" conda run --no-capture-output -n curve python -B -m exp6.search --plan exp6/search_plan.json --dry-run
```

Stage 2 search/freeze：

```bash
TMPDIR="$PWD/exp6/runtime/tmp" conda run --no-capture-output -n curve python -B -m exp6.search --plan exp6/search_plan.json --gpus 1,2,3
```

搜索 seed 固定为 20261001，选择严格按 **final_test_top1 最大、C 最小、lr 最小** 排序；不使用 loss、clipping/noise/geometry 诊断选型。只有 geom_eps 候选的 utility 和 C/LR 都相同时，追加较小 geom_eps 作为 deterministic tie break。

A：IID 先在 lr=1e-3 下搜索 C={0.1,0.3,1,3,10}，固定最好 C 搜 LR={1e-4,3e-4,1e-3,3e-3,1e-2,3e-2}，再做 C/lr 各乘 {0.5,1,2} 的 3×3 局部 refinement。

B：raw MF 用 IID winner 作 anchor，独立按 C→LR→局部 refinement 调参。

C：IID-scale 用 IID winner C/LR 搜 geom_eps={1e-3,1e-2,1e-1,1}，选定后冻结 geometry，独立搜索 C/LR 和局部 refinement。

D：MF-scale 复用 C 冻结的同一 geom_eps，以 IID-scale winner C/LR 为 anchor，只搜索自己的 C/LR 和局部 refinement。所有 BandInvMF trials 始终使用相同 T=250、beta1=.9、num_bands=4 和 momentum workload/matrix implementation。

局部 winner 在一个方向被邻居 bracket 时停止扩展。若位于局部区间边界，最多沿该方向再评估一个向外点；没有 utility improvement 就停止。整个过程是有限 staged/coarse-to-local 搜索，避免巨大 Cartesian grid。Stage summary 记录候选、scores、winner 和 stop reason。

每个 method 最终从其所有 staged completed candidates 选型；scale 在已冻结的 geometry 下选 C/LR。结果目录 `exp6/results/search/` 包括：

- `*_summary.json`、`*_stopping.json`：阶段候选、结果、winner、停止原因。
- `completed_trials.json`、`trials/<trial_identity>/`：全部完成 trials 的索引与完整 artifacts。
- `<method>_frozen.json`、`scale_geometry_frozen.json`、`selected_configs.json`：冻结配置、privacy/workload 与诊断。
- `provenance.json`、`sessions.json`、`search_summary.json`：计划、训练/搜索代码 hash、版本、资源指纹及训练/复用计数。
- `prepared_final_manifest.json`：12 个正式 trials 的准备清单；**不启动它们**。

`audit.py` 的 fingerprint 包含固定协议、所有 trial 参数、训练代码 SHA256、数据/权重 SHA256、依赖版本。复用时检查完整文件集、artifact hashes、250 个 steps/geometry records、test size=10000、epsilon/delta 和重新构造的策略矩阵。配置/代码/数据冲突或不完整目录直接失败，不自动修复、不重跑 completed 同一 trial。恢复时重新按阶段决策重放，只训练新的候选。

`search_summary.json.total_unique_tuning_trials` / `actual_trained_trials_total` 表示该搜索目录实际完成的不同 full trials 总数；`actual_trained_trials` / `reused_preexisting_trials` 表示本次 search invocation 新完成和启动前已完成的数量。`stage_reuse_events` 是在阶段间复用过的不同 trial 数，中心点等会复用而不重复训练。

本次搜索使用 official CIFAR-100 test top1 作为 utility selection metric，最终报告应按这个选择协议解释；test 不属于训练 privacy population。正式比较允许各方法有独立 utility-optimal C/LR，因此 factorial differences 是各方法调参后的 utility comparison。

参数冻结并人工检查后，显式启动正式实验（固定 seeds=20261011/12/13，四方法 × 三 seeds = 12 trials）：

```bash
TMPDIR="$PWD/exp6/runtime/tmp" conda run --no-capture-output -n curve python -B -m exp6.final_runner --selected exp6/results/search/selected_configs.json --gpus 1,2,3
```

正式入口会先核对四个 selected tuning trials 的 fingerprints。结果位于 `exp6/results/final/`，报告输出 `report.json` 和 `report.md`：mean final test top1、sample std (ddof=1)、每 seed accuracy、selected C/lr/geom_eps、privacy/workload/clipping/LoRA geometry metadata，以及 MF_gain_raw、MF_gain_scale、scale_gain_iid、scale_gain_mf。interaction=MF_gain_scale−MF_gain_raw，直接回答主研究问题；三 seeds 比较给出描述性结果和每 seed paired effects。
