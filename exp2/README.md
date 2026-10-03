# exp2: geometry × noise mechanism

CIFAR-100 / pretrained ViT-Tiny 全参数 FP32 实验。只修改 `exp2/`。
数据只读仓库 `data/`，`download=False`；预训练 checkpoint 只读仓库 `cache/`，
不下载。保持 5 epochs、logical batch=1000、physical batch=250、accumulation=4、
(k,b,T)=(5,50,250)，Adam betas=(0.9,0.999)、eps=1e-8、weight_decay=0。
隐私按固定 epoch GDP 校准：epsilon=8、delta=1e-5、add/remove zero-out、
无 sampling amplification。每个 logical batch 精确逐样本 clipping、
一次 private noise release、一次 Adam step。

Scale 使用 previous completed vhat：`s_t=1/(sqrt(vhat_{t-1})+0.1)`，
在该空间 clip/noise，除以同一步实际使用的 s 后交给 Adam。Standard 为 s=1。
6 factorial cells 为 IID、Prefix BandInvMF、Momentum BandInvMF 的 Standard/Scale。
factorization 只依赖 noise family、T、bands=4、beta1，geometry 不参与。
同 family 的 workload、strategy、noising coefficients 完全一致；clip 变化时
单独重新 GDP calibration，仅实际 `M=innovation_std_sum*D` 的标量改变。

固定 anchors：IID Standard (lr=5e-4,C=1)、Prefix Scale
(lr=2e-3,C=200,eps_scale=0.1)、Momentum Standard (lr=5e-3,C=1)。
search seed=20261001，按 final_test_top1 最大选择，tie 依次选较小 C、较小 lr。

1. Stage 1：IID/Momentum Scale 各搜索 C={50,100,200,300,500}，固定
   lr=2e-3、eps_scale=0.1；Prefix Standard C=1 搜索
   lr={5e-4,1e-3,2e-3,3e-3,5e-3,7e-3}。共 16 trials。
2. Stage 2：分别固定 Stage 1 胜出 C，IID Scale LR grid 为
   {5e-4,1e-3,2e-3,3e-3}，Momentum Scale 为
   {1e-3,2e-3,3e-3,5e-3,7e-3}。新增 7 trials，复用两个 lr=2e-3 点。
   stage2 CSV 有 9 行，复用行标记 `stage2_reused`。
3. Stage 3：以 Stage 2 固定 C 的 LR winner 为中心，分别取原 C/LR grids
   的当前值及存在的左/右邻居，生成局部 Cartesian grid，排除 Stage 1/2
   已完成点。IID 新增 1–4、Momentum 新增 1–6，合计 2–10 trials，
   实际数量由搜索结果决定。从 Stage 1+2+3 **全部**方法结果选择最佳配置，
   冻结 `selected_configs.json`。
4. Prefix Scale 固定 anchor 在 search seed 额外运行 1 个 reference trial；
   不参与 tuning。IID/Momentum Scale 复用全局胜出搜索 trial。
5. matched clipping：三个 family 取最终 Scale 的 epochs 3–5 clip_fraction
   均值为 target；Standard 使用相同 lr，搜索 C={1,3,10,30,100,300}，
   共 18 trials，各自重新 GDP calibration。按 epochs 3–5 mean clip fraction
   与 target 的绝对差最小选择，tie 选较小 C，冻结 `matched_configs.json`。
6. 所有配置确定后运行 seeds=20261011/12/13 的六个主 cell，共 18 trials。
   matched final 复用其中九个 Scale，只新增三个 matched Standard × 三 seeds，
   共 9 trials。完整 pipeline 共 `69 + Stage3` 个 unique trials（71–79）。

四个 worker 使用 GPU 0/1/2/3，每 trial 单 GPU，不用 DDP、Slurm 或 GNU parallel。
当前 stage 的空闲 GPU 立即取下一个 trial；每个 stage 完成才进入下一 stage。
stdout/stderr 写 trial `train.log`。任一 worker 失败则 drain 当前队列后非零退出，
后续 stage 不启动。不自动删除、修复或重跑失败 trial。
目录不存在则 launch；目录存在且通过 `read_completed()` 则 reuse；
缺失产物或未完成直接失败，提示：

```text
Incomplete trial directory: <path>
Remove it manually before restarting.
```

每个同 seed 方法共享 pretrained checkpoint、classifier/model initialization、
固定 training order 和 augmentation RNG convention。固定 permutation 每 epoch
重复分组；train/test DataLoader 分别使用独立 seeded generators；噪声 RNG 独立。
每个 logical batch 取前四张实际 transformed examples，在 CPU 更新 rolling SHA256，
每 epoch 重置并记录 `augmentation_trace_sha256`，不保存原始图像。
最终检查全部 factorial/matched 同 seed 的每 epoch hash、模型初值、顺序、坐标和
同 family MF factorization 完全配对。

固定 flattened parameter order；NumPy default_rng(0) 不放回选择并排序 2048 坐标。
全部方法/seeds 使用相同 indices。`mechanism_trace.npz` 保存：

- `coordinate_indices`: [2048]
- `p_trace`: [250,2048]，`p_t=1/(sqrt(vhat_t)+adam_eps)`，在 Adam.step 后读取
- `s_trace`: [250,2048]，本 step 实际使用的 previous-state Scale 或 Standard ones
- `r_trace`: [250,2048]，`r_t=p_t/s_t`

mechanism_metrics.csv 保留 r 分位数、mean/std、CV、p90/p10 anisotropy、
相邻时间 median absolute log drift；首步 drift 留空。smoke 的时间维度为 1。

`paper_approx_mf_distortion` 保留 `W diag(r) M` 的原诊断，使用实际保存的 W/M。
标记为 **paper-style approximation; not exact Adam Jacobian**。

frozen-v 使用 0-indexed `A[t,j]=(1-beta1)*beta1**(t-j)/(1-beta1**(t+1))`
（j<=t，否则 0），`L[t,j]=1[j<=t]`。对坐标 q：

```python
K_q = L @ diag(p_q) @ A @ diag(1/s_q) @ M
r_rms_q = sqrt(mean((p_q/s_q)**2))
K_ideal_q = r_rms_q * L @ A @ M
frozen_ratio_q = norm(K_q, 'fro') / norm(K_ideal_q, 'fro')
```

这是 **frozen-observed-v linearized noise operator**：固定观察到的 v 轨迹，
不包含 infinitesimal noise 对未来 v 的反馈，不是 exact Adam Jacobian。
实现用 Adam moment 和 prefix 的逐行 recurrence，计算同一 Frobenius norm，
无需存储 [2048,250,250] operator。tests 与显式矩阵乘法核对。
p 和 s 分别在时间上恒定时 frozen distortion=1；仅 r=p/s 恒定但 p、s
共同随时间变化时，A 与 diag(s) 一般不交换，公式不保证 distortion=1。

MF cancellation efficiency 的 baseline 定义为：

```python
K_iid_q = r_rms_q * L @ A @ (innovation_std_sum * I)
rmse_actual_q = norm(K_q, 'fro') / sqrt(T)
rmse_iid_q = norm(K_iid_q, 'fro') / sqrt(T)
efficiency_q = rmse_actual_q / rmse_iid_q
```

baseline 保持该坐标 effective RMS、同一 trial GDP 校准的 innovation std，以及
变换前 `T*innovation_std_sum**2` 创新能量；只把 temporal transform D 改为 identity。
不另作 IID calibration，也不匹配变换后 output energy。共同的 lr/logical_batch_size
在比值中抵消。**Lower = more effective temporal noise cancellation**。
此定义写入代码注释及 `mechanism_summary.json` metadata。

tuned 六个 cells 的 G/I 仍使用 Scale-minus-Standard 及 difference-in-differences，
输出明确标记 `tuned_system_effects`，各 cell 独立 tuning 的结果不能当成纯几何因果效应。
matched 输出称 `matched-clipping geometry comparison`，保留相同 LR 并匹配 late clipping，
减少混淆但不宣称严格因果效应。top1 以 fraction 表示，跨 seeds 使用 sample std(ddof=1)。
机制 step 指标先 trial 内均值再 seed 等权均值；ratio 汇总为 trial medians 的 median。
IID 的 MF fields 显式为 null。mechanism summary 包括六主 cells 和三个 matched cells。

完整实验生成：

```text
exp2/results/
  search_stage1_summary.csv
  search_stage2_summary.csv
  search_stage3_summary.csv
  search_summary.csv
  selected_configs.json
  matched_search_summary.csv
  matched_configs.json
  final_multiseed.csv
  final_summary.json
  factorial_effects.csv
  factorial_effects.json
  matched_multiseed.csv
  matched_effects.csv
  matched_effects.json
  paper_approx_mf_distortion.csv
  frozen_v_mf_distortion.csv
  mf_cancellation_efficiency.csv
  mechanism_summary.json
```

每 trial 保存 config.yaml、train.log、metrics.csv、summary.json、train_order.npy、
final.pt、matrices.npz、mechanism_trace.npz、mechanism_metrics.csv。
matched_multiseed.csv 包含九个原 Scale 路径及九个新 matched Standard 路径。
MF diagnostic CSV 包含主和 matched MF trials，并按 method/seed 区分。

正式实验前验证（curve 环境）：

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp2/cache/tmp" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 conda run --no-capture-output -n curve python -m pytest -c exp2/pytest.ini exp2/tests --basetemp=exp2/results/pytest_tmp
conda run --no-capture-output -n curve bash exp2/run_sweep.sh --smoke
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n curve python -m exp2.verify_smoke
```

六 cell smoke 各训练一个完整 logical batch（250×4），测试 100 张，仍按完整
250-step strategy 校准；不参与搜索、配置冻结或最终汇总。代码变更后只运行 tests
和 smoke，不自动启动正式实验。完整新实验的唯一启动命令：

```bash
conda run --no-capture-output -n curve bash exp2/run_sweep.sh
```
