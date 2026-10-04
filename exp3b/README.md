# Exp3b：utility continuation 与累计 optimizer-update cancellation

本目录中的实现读取 `exp2/`、`exp3/` 的已有配置与搜索结果，所有新代码、spec、运行缓存、测试产物、训练日志和结果均写入 `exp3b/`。数据只读取仓库 `data/`，预训练 checkpoint 只读取仓库 `cache/`；CIFAR-100 使用 `download=False`，timm 使用 `pretrained=False` 后显式读取本地 safetensors。没有联网、下载或自动恢复路径。

## 启动完整实验

在仓库根目录执行下面这一条命令：

```bash
source /home/longt29/miniconda3/etc/profile.d/conda.sh && conda activate curve && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m exp3b.experiment --gpus 0,1,2,3 --data-root data --cache-root cache
```

父进程通过独立 subprocess 为每个 trial 指定一张 GPU；每批最多四个进程，使用 GPU 0、1、2、3。任何新 trial 失败即终止当前批次并报错。已有完整且 spec 指纹匹配的 Exp3b trial 会复用；存在不完整或不匹配的目录会直接失败，由用户检查。

## 已读取的 utility 配置

| 方法 | Muon LR | Adam LR / 全模型 LR | C | active geometry |
|---|---:|---:|---:|---|
| nonprivate_hybrid | 0.0003 | 0.001 | 100（不 clipping） | identity |
| iid_dp_hybrid | 0.001 | 0.001 | 30 | identity |
| mf_muon_standard | 0.006 | 0.012 | 30 | identity |
| mf_muon_normscale | 0.006 | 0.012 | 30 | lambda_parallel=0.1 |
| mf_muon_spectralscale | 0.006 | 0.012 | 30 | kappa=1.1, rho=0.1 |
| momentum_standard / MF-Adam | — | 0.005 | 1 | ordinary clipping |
| momentum_scale / MF-Adam-Scale | — | 0.005 | 100 | eps_scale=0.1 |

Exp3 历史 selected Scale 配置是 identity；这里在 search history 中按 `final_test_top1` 选择最佳 **active** geometry。Standard 当前 utility=0.7905，active NormScale=0.7661，active SpectralScale=0.7764。Exp2 主配置来自 `selected_configs.json`，与 `final_summary.json` 一致；`momentum_scale` 同时与搜索 CSV 的最高 utility 配置核对。没有使用 matched Standard 或 cancellation 指标选配置。

当前 `results/selected_configs.json` 状态是 `pending_refinement`。本次实现阶段仅准备配置、计算精确 Momentum 理论、运行 unit tests 和极短 smoke；`final_summary.json` 标记 `pending_full_experiment`，不存在真实完整实验结果。

## 搜索与冻结

`specs/protocol.json` 明确记录搜索规则：固定 Standard Adam LR=0.012、C=30，从 Muon LR=0.009、0.012 顺序测试；第一处比此前最高 utility 低 **0.005**（0.5 个百分点）即停止。若尚无下降，按 4/3 增加 LR，最多增加四个点。仍未出现 upper bracket 时直接失败，并保留所有观测，不把未 bracket 的结果自动冻结。重复运行可复用已完成的配置。

每种 Scale 固定 Exp3 历史最佳 non-identity geometry，在最终 Standard Muon LR 的 0.75、1、1.25 倍附近进行局部 refinement，并补一个 Adam LR 的 0.75 倍配置。每种最多四个 LR 配置；完全相同的历史点复用。最终所有方法严格取最高 `final_test_top1`，然后冻结，运行 seeds 20261011、20261012、20261013。Smoke 分数不进入选择。

## Source 与 replay

MF-Adam、MF-Adam-Scale 各训练一条全模型 private source trajectory，seed=20261011。MF-Muon 直接复用最终三 seed 验证中 seed=20261011 的 Standard private trajectory，并在训练时捕获信号。每一步的 signal 文件均在 clipping 完成、noise 添加之前保存；训练仍使用原始真实 optimizer 和私有机制更新全模型。

保存与主 replay 的 support 为 12 个 blocks × 四类 matrix weights：`attn.qkv.weight`、`attn.proj.weight`、`mlp.0.weight`、`mlp.2.weight`，严格共 48 个参数。Adam 的逐坐标更新及 Muon 的逐矩阵更新允许仅保存这组 support；其余参数仍参与 source 的 clipping、噪声和真实训练，保证源轨迹不变。三种 nonlinear 方法使用完全相同的 support。

Adam-Scale 每步保存 frozen `S_t`、平均单位的 `q_t^S` 和 realized signal `g_t=q_t^S/S_t`。Replay clean 每步由 `q_t^S/S_t` 重建；MF/IID 分别为 `g_t + (Dz)_t/S_t`、`g_t + z_t/S_t`，不重算 Scale。Raw gradient 不参与 RMSE，因此 clipping bias 不进入误差。

Replay 不调用 forward/backward，也不更新 source model 或 signal 文件。clean、MF、IID 的 optimizer state 从零开始并独立演化。MF/IID 使用同一批 Gaussian tensor `z`；每步只生成一次创新，MF 做 FIR filter、IID 直接使用同一个 `z`。创新标准差为 source 的 `innovation_std_sum / logical_batch_size`，IID 不进行新 privacy calibration。默认 paired seeds 为 20261101–20261108。

Adam 完整 update 为 bias-corrected `mhat/(sqrt(vhat)+1e-8)`。Muon 完整 update 使用实际 Nesterov beta=0.95、Frobenius normalization、NS5 和 shape factor。MF workload 始终是 ordinary momentum beta=0.9、num_bands=4、T=250。

主 metric：`delta_u_t = u_noisy_t - u_clean_t`，`E_t=sum(delta_u_j)`，`RMSE_t=sqrt(sum(E_t²)/d)`，均为 **pre-LR 完整 update direction**；每步同时输出 instantaneous update RMSE。`C_t=RMSE_MF/RMSE_IID` 在每个 paired seed 上计算，再报告 mean 和 sample std（ddof=1）。`relative_to_momentum=mean(C_t)/C_t_momentum`。乘 source 实际 LR 的 parameter-space RMSE 只保存为 secondary metric。

Momentum 不训练也不采样：`W=L H_0.9`，逐步精确计算 `||WD[t,:]||/||W[t,:]||`。其 RMSE 使用 unit innovation scale，图中明确标注；跨算法可直接解释的相对量是 cancellation 和 relative-to-momentum。

## 输出与目录

```text
exp3b/
  __init__.py, spec.py, support.py
  history.py, experiment.py, scheduler.py
  train.py, capture.py, replay.py, report.py, smoke.py
  README.md, pytest.ini
  specs/
    protocol.json, standard_0.009.json, standard_0.012.json
    generated/                         # 逐 trial 的 exact specs
  tests/
    conftest.py, test_cancellation.py, test_planning.py
  runtime/                             # 所有运行缓存、临时文件
  results/
    input_audit.json, selected_configs.json, tuning_summary.csv
    final_summary.json, standard_bracket.json
    unit_tests.log, prepare.log, smoke.log, smoke_verification.json
    launch_logs/
    smoke/                             # 7 种机制，每种 2 个极小 logical steps
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

测试覆盖理论与小型 Monte Carlo、250 步齐全且 finite、Gaussian pairing、独立 Adam state、Scale realized clean signal/noise-only inversion/clipping bias 排除、实际 Adam/Muon update 一致性、48 参数 support、source 不变、utility-only 选择、历史复用与顺序 stopping。Smoke 在 GPU 0–3 用 dim=8、12 blocks、32×32 的小模型运行每种机制两个 logical steps，读取真实本地 CIFAR-100 并验证本地 checkpoint 可加载；三种 source 各用八个 paired seeds 做两步 replay 和绘图。
