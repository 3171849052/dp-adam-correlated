# Exp3b：utility continuation 与累计 optimizer-update cancellation

本目录中的实现读取 `exp2/`、`exp3/` 的已有配置与搜索结果，所有新代码、spec、运行缓存、测试产物、训练日志和结果均写入 `exp3b/`。数据只读取仓库 `data/`，预训练 checkpoint 只读取仓库 `cache/`；CIFAR-100 使用 `download=False`，timm 使用 `pretrained=False` 后显式读取本地 safetensors。没有联网、下载或自动恢复路径。

## 启动完整实验

在仓库根目录执行下面这一条命令：

```bash
source /home/longt29/miniconda3/etc/profile.d/conda.sh && conda activate curve && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m exp3b.experiment --gpus 0,1,2,3 --data-root data --cache-root cache
```

父进程通过独立 subprocess 为每个 trial 指定一张 GPU；每批最多四个进程，使用 GPU 0、1、2、3。任何新 trial 失败即终止当前批次并报错。已有完整且 spec 指纹匹配的 Exp3b trial 会复用；存在不完整或不匹配的目录会直接失败，由用户检查。

后台启动完整流程（同样使用 conda `curve`、GPU 0–3、只读 `data/` 和 `cache/`）：

```bash
bash exp3b/launch_full.sh
```

该命令使用 `nohup setsid`，记录 `results/full_experiment.pid`、`background_launch.json`，将主流程日志写入 `results/full_experiment.log`。已有 PID 文件时直接失败，防止重复提交完整流程。

## 已读取的 utility 配置

| 方法 | Muon LR | Adam LR / 全模型 LR | C | active geometry |
|---|---:|---:|---:|---|
| nonprivate_hybrid | 0.0003 | 0.001 | 100（不 clipping） | identity |
| iid_dp_hybrid | 0.001 | 0.001 | 30 | identity |
| mf_muon_standard | 0.006 | 0.012 | 30 | identity |
| mf_muon_normscale | 0.006 | 0.012 | 30 | lambda_parallel=0.1 |
| mf_muon_spectralscale | 0.006 | 0.012 | 30 | kappa=1.1, rho=0.1 |
| momentum_standard / MF-Adam 当前 history best | — | 0.005 | 30 | ordinary clipping；继续 Exp3b utility tuning |
| momentum_scale / MF-Adam-Scale | — | 0.005 | 100 | eps_scale=0.1 |

Exp3 历史 selected Scale 配置是 identity；这里在 search history 中按 `final_test_top1` 选择最佳 **active** geometry。Muon Standard 当前 utility=0.7905，active NormScale=0.7661，active SpectralScale=0.7764。

Exp2 的 MF-Adam Standard `lr=.005,C=1` 是 fixed anchor，不是 utility-tuned optimum。现在读取 `exp2/results/search_summary.csv` 和 `matched_search_summary.csv`，对每个真实完成的 `momentum_standard` / `momentum_standard_matched` trial 验证 summary、seed、250 steps、LR、C 和 `final_test_top1`；将它们统一作为同一 Momentum BandInvMF + Adam Standard mechanism 的 utility history，保留原始 method 标签和 CSV provenance。当前最高观测为 `lr=.005,C=30,top1=.7361`（来自 matched-search），超过 C=1 的 .7235。唯一排序依据是 utility，匹配 clipping 的指标不参与选择，matched 标签没有优先权。

MF-Adam-Scale 从 Exp2 `momentum_scale` 的所有 completed utility observations 中取最大 utility；当前仍为 `lr=.005,C=100,eps_scale=.1,top1=.745`。若历史中已有更高 utility，选择更高者。Exp2 原 selected configs 与 final summary 仍作为历史记录保存在 audit，Standard fixed anchor 不再用作主 source 配置。

`--prepare` 仅刷新 history audit、pending selected configs、tuning summary、起始 specs 和精确 Momentum 理论，不训练。`results/selected_configs.json` 在 continuation 完成前为 `pending_refinement`，完成后为 `frozen`。`results/background_launch.json`、`full_experiment.pid` 与 `full_experiment.log` 记录后台启动；训练和 cancellation 完成情况分别见各 trial summary 与最终 cancellation summary。

## 搜索与冻结

MF-Adam 自己的 utility continuation 先执行。固定 history best 的 C（当前 30），围绕 .005 测试 .003、.007；相同已完成配置直接复用。若已有低 LR 和高 LR 观测均比当前该 C 的最高 utility 下降超过 .005，则 LR 已 bracket，立即停止。需要延伸时仅补缺失的一侧：更低 LR 除以 1.5，更高 LR 乘以 1.5，最多三个额外点；无双侧 bracket 即 assert/fail。若 bracket 最优 LR 相对 .005 移动至少 20%，仅在该 LR 比较 C=10/30/100（C=30 直接复用，最多新增两个 clipping trials）。最终从全部历史和新增观测中取严格最大 `final_test_top1`，不使用 cancellation 或 matched-clipping 指标。

`results/mf_adam_continuation.json` 保存双侧 bracket、每个选择/复用点、最终配置及 provenance。`selected_configs.json` 的 `mf_adam_tuning` 和 `provenance.momentum_standard`、`input_audit.json`、`tuning_summary.csv` 记录完整来源。**MF-Adam uses its own utility-optimal hyperparameters selected by Exp3b continuation.** MF-Adam source 的 LR/C 从最终 frozen selection 读取。

`specs/protocol.json` 明确记录搜索规则：固定 Standard Adam LR=0.012、C=30，从 Muon LR=0.009、0.012 顺序测试；第一处比此前最高 utility 低 **0.005**（0.5 个百分点）即停止。若尚无下降，按 4/3 增加 LR，最多增加四个点。仍未出现 upper bracket 时直接失败，并保留所有观测，不把未 bracket 的结果自动冻结。重复运行可复用已完成的配置。

每种 Scale 固定 Exp3 历史最佳 non-identity geometry，在最终 Standard Muon LR 的 0.75、1、1.25 倍附近进行局部 refinement，并补一个 Adam LR 的 0.75 倍配置。每种最多四个 LR 配置；完全相同的历史点复用。最终所有方法严格取最高 `final_test_top1`，然后冻结，运行 seeds 20261011、20261012、20261013。Smoke 分数不进入选择。

## Source 与 replay

MF-Adam、MF-Adam-Scale 各训练一条全模型 private source trajectory，seed=20261011。MF-Muon 直接复用最终三 seed 验证中 seed=20261011 的 Standard private trajectory，并在训练时捕获信号。每一步的 signal 文件均在 clipping 完成、noise 添加之前保存；训练仍使用原始真实 optimizer 和私有机制更新全模型。

保存与主 replay 的 support 为 12 个 blocks × 四类 matrix weights：`attn.qkv.weight`、`attn.proj.weight`、`mlp.0.weight`、`mlp.2.weight`，严格共 48 个参数。Adam 的逐坐标更新及 Muon 的逐矩阵更新允许仅保存这组 support；其余参数仍参与 source 的 clipping、噪声和真实训练，保证源轨迹不变。三种 nonlinear 方法使用完全相同的 support。

Adam-Scale 每步保存 frozen `S_t`、平均单位的 `q_t^S` 和 realized signal `g_t=q_t^S/S_t`。Replay clean 每步由 `q_t^S/S_t` 重建；MF/IID 分别为 `g_t + (Dz)_t/S_t`、`g_t + z_t/S_t`，不重算 Scale。Raw gradient 不参与 RMSE，因此 clipping bias 不进入误差。

Replay 不调用 forward/backward，也不更新 source model 或 signal 文件。clean、MF、IID 的 optimizer state 从零开始并独立演化。MF/IID 使用同一批 Gaussian tensor `z`；每步只生成一次创新，MF 做 FIR filter、IID 直接使用同一个 `z`。创新标准差为 source 的 `innovation_std_sum / logical_batch_size`，IID 不进行新 privacy calibration。默认 paired seeds 为 20261101–20261108。

Adam 完整 update 为 bias-corrected `mhat/(sqrt(vhat)+1e-8)`。Muon 完整 update 使用实际 Nesterov beta=0.95、Frobenius normalization、NS5 和 shape factor。MF workload 始终是 ordinary momentum beta=0.9、num_bands=4、T=250。

主 metric 使用 **pre-LR 完整 update direction**：`delta_u_t = u_noisy_t - u_clean_t`，`E_t=sum(delta_u_j)`。每个 replay seed/step 保存 `cumulative_mse_mf`、`cumulative_mse_iid`、`instant_mse_mf`、`instant_mse_iid`（完整 support 的平方误差除以 d），以及 secondary parameter MSE。

正式跨 seed 定义为：

```text
RMSE_t = sqrt(mean_seed(||E_t||^2 / d))
C_t = RMSE_t^MF / RMSE_t^IID
relative_to_momentum = C_t / C_t^Momentum
instant_RMSE_t = sqrt(mean_seed(||delta_u_t||^2 / d))
```

主表字段为 `rmse_mf`、`rmse_iid`、`cancellation`、`relative_to_momentum`，以及 `instant_rmse_*` 和 secondary `parameter_rmse_*`。先平均 MSE，再开根号；正式 cancellation 是两个正式 RMSE 的比值。

**per-seed realized RMS and paired cancellation ratios are auxiliary diagnostics only**。`*_paired_samples.csv` 中保存每个 seed 的 `realized_rmse_*` 和 `paired_cancellation`。主表中的 `realized_rmse_mf_std`、`realized_rmse_iid_std` 及相应 instantaneous/parameter std 为每 seed realized RMS 的 sample std（ddof=1）；`paired_cancellation_mean/std` 是每 seed ratio 的辅助统计。它们不是正式 RMSE/cancellation estimator 的标准误或 CI。图只展示正式曲线，不把这些辅助 std 画成 estimator uncertainty。乘 source 实际 LR 的 parameter-space RMSE 仍只是 secondary metric。

Momentum 不训练也不采样：`W=L H_0.9`，逐步精确计算 `||WD[t,:]||/||W[t,:]||`。其 RMSE 使用 unit innovation scale，图中明确标注；跨算法可直接解释的相对量是 cancellation 和 relative-to-momentum。

## 输出与目录

```text
exp3b/
  __init__.py, spec.py, support.py
  history.py, experiment.py, scheduler.py
  train.py, capture.py, replay.py, report.py, smoke.py
  README.md, pytest.ini, launch_full.sh
  specs/
    protocol.json, standard_0.009.json, standard_0.012.json
    adam_standard_0.003.json, adam_standard_0.007.json
    generated/                         # 逐 trial 的 exact specs
  tests/
    conftest.py, test_cancellation.py, test_planning.py
  runtime/                             # 所有运行缓存、临时文件
  results/
    input_audit.json, selected_configs.json, tuning_summary.csv
    final_summary.json, standard_bracket.json, mf_adam_continuation.json
    unit_tests.log, prepare.log, smoke.log, smoke_verification.json
    launch_logs/
    smoke/                             # 保留旧版 smoke artifacts，旧聚合不是正式结果
    smoke_v2_utility_rms/               # 当前 smoke：MF-Adam C=30、正式 root(mean(MSE))
      cancellation/                    # 2-step smoke CSV/JSON 与 PNG/PDF
    tests/                             # unit test 的全部临时产物
    tuning/                            # 完整启动后生成
    final/<method>/seed_<seed>/         # 完整启动后生成；Muon 第一个 seed 有 signals/
    source/<method>/signals/           # 完整启动后生成；Adam/Adam-Scale
    cancellation/
      momentum_theory.csv              # 已生成，真实精确 250 步理论
      adam.csv, adam_scale.csv, muon.csv
      combined.csv, summary.json
      *_paired_samples.csv, adam.json, adam_scale.json, muon.json
      cumulative_rmse.png/pdf
      cancellation_curve.png/pdf
      relative_to_momentum.png/pdf
```

完整训练型 cancellation CSV 和图在完整启动后生成。`signals/step_NNN.pt` 每步独立写入，`manifest.json` 记录参数 support、shape、innovation scale、source spec、checkpoint 和各文件 SHA256。Replay 逐步读取并检查校验和，不把全部 trajectory 装入 GPU。

## 仅运行验证

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 conda run -n curve --no-capture-output python -m pytest -c exp3b/pytest.ini exp3b/tests -q
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 conda run -n curve --no-capture-output python -m exp3b.smoke
```

测试覆盖理论与小型 Monte Carlo、250 步齐全且 finite、Gaussian pairing、独立 Adam state、Scale realized clean signal/noise-only inversion/clipping bias 排除、实际 Adam/Muon update 一致性、48 参数 support、source 不变、MF-Adam history 读取/复用/utility-only bracket/最优配置传递给 source，以及两种不同 magnitude 的 seeds 下正式 `sqrt(mean(MSE))` 与辅助 mean RMS/mean paired ratio 的区别。Smoke 在 GPU 0–3 用 dim=8、12 blocks、32×32 的小模型运行每种机制两个 logical steps，读取真实本地 CIFAR-100 并验证本地 checkpoint 可加载；三种 source 各用八个 paired seeds 做两步 replay 和绘图。新 smoke 写入 `results/smoke_v2_utility_rms/`，保留旧版结果供追溯。
