# exp4：Update Clip DP-Adam

这是可独立运行的 CIFAR-100 + pretrained ViT-Tiny 实验平台，仅有
`dp-adam-iid-uc` 和 `dp-adam-bandinvmf-uc`。不导入其他 `exp*/` 的代码。
所有代码、配置、日志、输出和运行缓存都在 `exp4/`；数据从仓库 `data/`
读取，预训练权重从仓库 `cache/` 读取。使用 conda 环境 `curve`。

## UC 定义与执行顺序

四个 physical minibatch 的普通 cross-entropy 梯度累积为一个 logical
minibatch 的平均梯度：每个 minibatch 使用 `loss.sum()/1000` 反向传播。
没有逐样本梯度计算、逐样本 clipping、梯度噪声或 Opacus wrapper。

`optimizer_uc.py` 的 `UCAdam.step()` 明确按以下顺序执行：

\[
m_t=0.9m_{t-1}+0.1g_t,\qquad
v_t=0.999v_{t-1}+0.001g_t^2,
\]
\[
\hat m_t=m_t/(1-0.9^t),\quad \hat v_t=v_t/(1-0.999^t),\quad
u_t=\hat m_t/(\sqrt{\hat v_t}+10^{-8}),
\]
\[
q_t=u_t\min(1,R/\|u_t\|_2),\qquad
\widetilde u_t=q_t+n_t,\qquad
\theta_{t+1}=\theta_t-\eta\widetilde u_t.
\]

L2 norm 对所有 trainable parameters 的 direction 拼接后计算，只使用一个
全局 clip scale。实现逐 tensor 求平方和，等价于 flatten 后做一次 global
clipping。`u_t` 不含学习率、不含负号。学习率和负号只在最后的 parameter
update 出现；不调用 `torch.optim.Adam.step()`。`m/v` 只接收当前真实模型位置
上的 raw mean batch gradient，clip 和 DP noise 不写入隐藏状态。此前的 noisy
parameter update 会通过下一步的真实梯度影响状态，这是正常的训练反馈。

两个方法共用 `train.py`、`UCAdam` 和 `TemporalGaussianNoise`；唯一差别是
temporal filter 及其相应隐私校准。`noise_seed = trial_seed + 1`。

## Temporal noise 与隐私校准

固定 horizon `T=250`，按 replace-one 邻接采用每步保守敏感度 `2R`。
即使一个相邻样本及 raw Adam 状态影响全部 250 个 direction，每个相邻
direction 的差也至多为 `2R`。不使用 sampling amplification、不除以 batch
size、不使用 `j,j+b,...` 或 fixed-epoch sparse participation accountant。

令 `D=C^{-1}`，`z_t ~ N(0, sigma_z² I)` 独立，`n=Dz`。

- IID：`D=C=I`，每步独立 Gaussian noise。
- BandInvMF：复制/整理 `exp2/bandinvmf.py` 中的 Toeplitz 数学构造。
  workload coefficients 固定为全 1，workload matrix `W` 为下三角全 1 的
  SGD/prefix-sum matrix；不使用 momentum workload。固定 bandwidth=4，得到
  `d=[1,-1/2,-1/8,-1/16]`。在线输出
  `n_t=sigma_z * sum_l d_l e_{t-l}`，保存的是独立 innovations，最多 3 步历史。
  `C` 是完整 250×250 下三角 `D` 的逆。

对依赖此前 noisy 参数的 adaptive UC queries，给定相同的已发布历史，
whitened 第 t 步的条件 Gaussian mean difference 保守界为
`2R * sum_j abs(C[t,j])`。将这些 conditional Gaussian bounds 做 GDP 组合：

\[
S(C)=\sqrt{\sum_t\left(\sum_j|C_{tj}|\right)^2},\qquad
\mu=\frac{2R S(C)}{\sigma_z}.
\]

本实验 `C` 元素非负，因此此界也等于全参与 aligned-direction 的
`sqrt(sum(C.T @ C))`；无需假设各步方向相同。对可能含负元素的策略仍保留
行绝对值和界，以覆盖 adaptive conditional means。

`privacy.py` 解以下方程获取目标 GDP `mu`，然后取
`sigma_z=2R*S(C)/mu_target`：

\[
\delta=\Phi(-\epsilon/\mu+\mu/2)
       -e^\epsilon\Phi(-\epsilon/\mu-\mu/2).
\]

`epsilon=8, delta=1e-5` 时，`mu_target≈1.666030598`。
IID 的 `S(I)=sqrt(250)`，所以
`sigma_iid=2R*sqrt(250)/mu_target≈18.98090986R`，严格对应 250 个独立机制的组合。
bandwidth=4 时 `S(C)≈50.10539577`，`sigma_z≈60.14943043R`。
BandInvMF 的 step marginal noise std 为
`sigma_z*sqrt(sum_{l=0}^{min(t,3)} d_l²)`，会在开始几步变化。
这些都是 **update-direction space** 的每 coordinate 标准差。
参数空间标准差为 `lr * noise_marginal_std`。

每个 epoch 的已用 GDP bound 使用同一完整策略的前若干行及相同完整 horizon
校准的 `sigma_z`；不会在中途重新校准。smoke 也按完整 250 步校准。
完整 trial 的目标为 `(8,1e-5)`，此目标不代表整个超参搜索的总隐私预算。
隐私声明覆盖 noisy 参数轨迹及其对公开 test 数据的评估；保存的 raw
direction、Adam 状态统计、train loss、clip 统计和数据顺序属于内部研究诊断，
并未额外做 DP 处理。发布这些诊断不包含在上述隐私保证中。

## 固定实验协议

| 项目 | 固定值 |
|---|---|
| 数据 / 模型 | CIFAR-100 / ViT-Tiny patch16 224，全部参数训练 |
| epochs / optimizer steps | 5 / 250 |
| logical / physical batch | 1000 / 250 |
| gradient accumulation | 4 |
| Adam beta1 / beta2 / eps / weight_decay | 0.9 / 0.999 / 1e-8 / 0 |
| epsilon / delta | 8 / 1e-5 |
| BandInvMF workload / bandwidth | SGD prefix / 4 |
| search seed | 20261001 |
| final seeds | 20261011、20261012、20261013 |
| GPU / 并行上限 | physical 0、2、3 / 3 个 trial |

学习率固定不衰减，仅搜索 `lr` 和 `update_clip_norm=R`。配置校验拒绝修改
上述固定协议和 UC 语义字段。训练不使用 AMP 或 DDP，每个 subprocess 只暴露
一个允许的 physical GPU，进程内使用 `cuda:0`。JAX 构造系数时只使用 CPU。

保持与 `exp2` 相同的模型 tensor mapping、随机初始化 100-way head、224 像素
训练增强及 test preprocessing：RandomResizedCrop + HorizontalFlip；test 使用
Resize + CenterCrop；按 pretrained metadata normalize。每 seed 只生成一次
训练 permutation，各 epoch 复用，同 seed 两个方法共用初始模型、数据顺序和
增强 RNG 协议。开启 deterministic algorithms，关闭 TF32。

固定使用 FP64 Adam m/v、bias correction 和 direction 算术，随后将 direction
转换为 FP32 再做 UC/noise/parameter update。模型参数、普通累积梯度、UC 查询和
Gaussian noise 均为 FP32；LayerNorm 和 attention 内部计算固定为 FP64，
eps/参数/结构保持相同。固定使用 math SDPA 后端，在训练时对每个 Transformer block 做 activation
checkpointing，保留 physical batch=250。没有 AMP、loss scaling 或额外梯度裁剪。
这些是所有 probe/search/final trial 共用的固定数值实现，不参与调参。

指定的大 R 曾触发 FP32 vhat 溢出、efficient SDPA 的错误巨大梯度/NaN，
以及 FP32 math attention 的 SafeSoftmaxBackward0 NaN；
相关尝试及 anomaly traces 在 `results/search_archives/`、`results/*nan_debug.log`。
旧精度结果全部排除排名，probe 和全部候选重跑。当前实现通过 26 项测试，
数值复现检查（`results/precision_preflight.json`、`results/attention64_preflight.json`）
仅用于修复验证，位于 runtime/，不作为搜索 trial 或用于超参选择。

`model.py` 显式读取
`cache/huggingface/hub/models--timm--vit_tiny_patch16_224.augreg_in21k_ft_in1k/refs/main`
指向的本地 `model.safetensors`，校验实际 blob 仍在仓库 `cache/` 内，并记录 SHA256。
`timm.create_model(..., pretrained=False)` 后严格加载该 checkpoint。
`CIFAR100(..., download=False)` 从 `data/cifar-100-python/` 读取。无下载路径，
缺少文件直接失败。runtime 设置 HF/Transformers offline 和 `exp4/runtime/` 缓存位置。

## 目录与每 trial 的输出

```text
exp4/
  config.yaml, config.py     固定协议与校验
  model.py                  本地 pretrained ViT-Tiny
  optimizer_uc.py            raw Adam -> UC -> noise -> lr
  bandinvmf.py               SGD workload 的 Toeplitz 数学
  privacy.py, mechanism.py   全参与 GDP 校准和在线 temporal noise
  train.py                  单 GPU shared training
  launch_batch.py            GPU 0/2/3 FIFO trial 队列
  search.py                 严格 Stage 0–6 顺序搜索，复用完整点并冻结
  final_runner.py            frozen config × 三 final seeds
  report.py                 trial CSV 和跨 seed 汇总
  verify_smoke.py            两方法 smoke 结果核对
  specs/                    staged search protocol、probe 和 smoke spec
  tests/                    数值、机制、本地资源和协议测试
  results/                  所有日志、trial 结果、汇总
  runtime/                  临时文件、库缓存、pytest 缓存与测试产物
```

每 trial 保存 `config.yaml`、`steps.csv`、`metrics.csv`、`summary.json`、
`matrices.npz`、`train_order.npy`、`final.pt` 和 launcher 写入的 `train.log`。
`final.pt` 仅保存模型参数和 step count。原始 Adam 状态不作为可恢复 checkpoint
发布；其基本统计写入诊断表。无自动恢复或兼容旧结果逻辑，要求使用新 trial 目录。

`steps.csv` 包含 raw `||u||`、clipped `||q||`、clip scale、step clip indicator、
raw/clipped update RMS、m norm、vhat min/max/mean/RMS、每步 train loss、
noise marginal std、parameter noise std、signal parameter norm、预期和实际
noise/signal RMS ratio。预期 ratio 定义为
`noise_marginal_std / (||q||/sqrt(number_of_parameters))`；实际 ratio 使用
这一步生成的 noise RMS。零信号的 ratio 留空，因为其比值无定义。

`metrics.csv` 记录每 epoch 的 train/test loss、test top1（0–1）、
**按 logical steps 计数**的 epoch clip fraction、已用 GDP mu/epsilon、
末步 noise std、耗时和增强 fingerprint。`config.yaml` 记录全部 privacy
calibration 参数及用户要求的 UC 语义字段；`matrices.npz` 保存 `D/C/W`，可审计
workload 和 filter inversion。

## 阶段 1 验证

在仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp4/runtime/tmp" conda run --no-capture-output -n curve python -B -m pytest exp4/tests -q -o cache_dir=exp4/runtime/pytest_cache --basetemp=exp4/runtime/tests > exp4/results/unit_tests.log 2>&1
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp4/runtime/tmp" conda run --no-capture-output -n curve python -B -m exp4.launch_batch --spec exp4/specs/smoke.json > exp4/results/smoke_launcher.log 2>&1
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp4/runtime/tmp" conda run --no-capture-output -n curve python -B -m exp4.verify_smoke > exp4/results/smoke_verification.log 2>&1
```

单元测试覆盖 UC 次序、全模型 clipping、bias correction、隐藏状态不接收噪声、
四次反传等于 mean batch gradient、SGD workload、full temporal sensitivity、
bandwidth=1 / identity 下 IID 的逐位一致性、FIR innovations、经验 covariance、
三 GPU 队列和 frozen trial/report 协议。真实本地 CIFAR 和 pretrained checkpoint
加载测试会禁止 socket network connections。

smoke 每方法仅 2 个 logical steps（8 个 250-example physical batches），
评估 100 个本地 test 样本；两个方法各自单 GPU 运行，保留完整 horizon 校准。
核对初始模型、classifier、数据顺序、增强 fingerprint 和所有诊断。
smoke accuracy 不用于挑选配置。完整搜索和最终实验不属于阶段 1。

已完成阶段 1 验证：`results/unit_tests.log` 中 **19 passed**；GPU 0 上 IID
和 GPU 2 上 BandInvMF 各完成 2 步，所有 loss/诊断为有限值。
`results/smoke_verification.json` 核对通过本地资源、prefix workload、UC
诊断、paired initialization/order/augmentation 以及第一步加噪前完全相同的
Adam/UC 统计。训练 GPU 已释放；没有启动完整搜索或 final trials。

## 严格 Stage 0–6 自动调参

当前搜索按以下顺序执行，仅改变 lr 和 R，不使用旧 `search_grid.json`。

| Stage | 方法 / 目的 | 候选 |
|---|---|---|
| 0 | nonprivate scale probe | seed=20261001, lr=1e-3，无 UC clipping，无 noise；250 步 norm 中位数为 U50 |
| 1 | IID，固定 lr=1e-3 | R/U50 = 1/256,1/128,1/64,1/32,1/16,1/8,1/4,1/2,1 |
| 2 | IID，固定 Stage 1 最优 R | lr = 2.5e-4,5e-4,1e-3,2e-3,4e-3 |
| 3 | IID，以 Stage 1+2 最佳点为中心 | R 和 lr 各乘 1/2,1,2，共 3×3 点 |
| 4 | BandInvMF，固定 IID 最终最优 lr | 与 Stage 1 相同的九个 R/U50 |
| 5 | BandInvMF，固定 Stage 4 最优 R | lr 为 IID winner lr 的 1/4,1/2,1,2,4 倍 |
| 6 | BandInvMF，以 Stage 4+5 最佳点为中心 | R 和 lr 各乘 1/2,1,2，共 3×3 点 |

所有 trial，包括 probe，都完整跑 5 epochs / 250 steps。probe 使用相同
raw-gradient Adam，仅设 `update_clip_norm=None` 和 `ZeroNoise`，不生成 Gaussian
随机数；明确记录 nonprivate，不校准或声明 GDP。其最终准确率不能用于排名。

每方法最终从该方法三个搜索阶段的全部完整 trial 中，以最后一个 epoch
的 `final_test_top1` 最大值选取；tie 时先选更小 R，再选更小 lr。
**test loss 不用于打破 tie**。Stage 3/6 的中心也是此前全部该方法候选的最佳点。
Stage 3 完成后先冻结 IID 的中间配置，然后开始 Stage 4；全部完成后只冻结两方法，
不会启动 final seeds。

相同 `(method, seed, lr, R)` 以 float hex 和 SHA256 标识。仅当 `summary.json`
存在且完整 epoch/steps/seed/privacy 协议校验通过时复用；smoke、probe 不参与
DP 点的复用或排名。已有不完整目录直接失败，不自动修复、删除或重跑。
`specs/staged_search.json` 保存固定协议；每个候选的完整 spec 保存到
`results/search/specs/`。完成每个 trial 后立即更新搜索记录。

输出：

- `results/search_summary.csv`：按阶段/候选记录 reuse、final top1、全 250 步
  clip fraction、最后 epoch clip fraction、R、R/U50、lr 和 calibration。
- `results/search_history.json`：完整 specs、阶段结果、epoch metrics、复用与选择理由。
- `results/search_report.md`：阶段汇总与两方法选择理由。
- `results/selected_configs.json`：两方法冻结配置及 winner 的校准和来源。
- `results/search/scale_probe.json`：250 个 raw norms 和 U50。
- `results/staged_search.log`：调度与阶段日志；每 trial 的 `train.log` 在 trial 目录。

从仓库根目录启动（已存在的完整点可复用）：

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp4/runtime/tmp" conda run --no-capture-output -n curve python -B -m exp4.search > exp4/results/staged_search.log 2>&1
```

搜索是 test-based tuning，完整 trial 的 epsilon=8 是每 trial 的预算；整个搜索
含 nonprivate probe，不能解释为一次 epsilon=8 的私有训练。

## 冻结配置的最终三 seed 验证（搜索完成后手动启动）

两个方法 × seeds 20261011、20261012、20261013，共六个完整 trial，
每个单 GPU，仅用 physical 0、2、3，最多三个并行：

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp4/runtime/tmp" conda run --no-capture-output -n curve python -B -m exp4.final_runner --frozen exp4/results/selected_configs.json --result-root exp4/results/final > exp4/results/final_launcher.log 2>&1
```

输出 `results/final/final_multiseed.csv`、`results/final/final_summary.json`，
含各 seed 的 final test top1、每方法 mean test top1 和 sample std（ddof=1）。
同时保存 frozen config 副本、六个 trial、`trials.csv`、`aggregate.csv` 和 `report.md`。
final 验证仍使用同一个 test 集，不是独立 holdout 的泛化估计。
