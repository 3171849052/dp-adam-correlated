# exp2: Scale geometry × noise mechanism

独立 CIFAR-100 / ViT-Tiny 全参数 FP32 实验。六个 cell 为 IID、Prefix
BandInvMF、Momentum-workload BandInvMF 各自配对 Standard/Scale geometry。
`exp1*` 不被修改。数据只从仓库 `data/` 读取，`download=False`；模型只从
仓库 `cache/` 的固定 timm HF snapshot 读取。缺失 checkpoint 直接失败。
沿用 exp1e 的严格 backbone 映射，保留按 seed 初始化的 100 类 classifier。
各 trial 的 config/summary 记录 checkpoint、初始模型和 classifier 的 SHA256。

固定 5 epochs、logical batch=1000、physical batch=250，自动导出 accumulation=4、
(k,b,T)=(5,50,250)。固定同一 seed 的训练 permutation，每 epoch 重用同一分组。
这样参与间隔为 50，不作 sampling amplification。隐私使用固定 epoch GDP 校准，
add/remove zero-out、epsilon=8、delta=1e-5；任何 clip 都单独重新校准 innovation std。
每个 logical batch 精确逐样本 clipping、一次 private noise output、一次 Adam step。
Scale 在 logical batch 开始冻结 `1/(sqrt(vhat_previous)+0.1)`，在该空间
clip/noise，然后除以实际冻结的 scale 再交给 Adam。

同一噪声行 Standard/Scale 的 factorization 函数只接受噪声类型，不接受 geometry。
Prefix workload 系数为 ones；Momentum workload 完全沿用 exp1e 的
`toeplitz.multiply(ones, beta1**arange(T))`。trial 的 `matrices.npz` 存储
factorization、workload coefficients、`W` 和实际 `M=innovation_std*D`。
clip 会改变 `M` 的校准标量，始终保留相同 `D=C^-1`。

搜索 seed=20261001；三个 anchor 固定。搜索 23 个 unique trials：
IID Scale 5 clips + 3 个新增 LR；Prefix Standard 6 LR；Momentum Scale
5 clips + 4 个新增 LR。第一阶段共 16 个 trial；全部完成后选择两行最佳 clip，
第二阶段运行 7 个新增 LR，复用 lr=2e-3 的 clip winner。按 final_test_top1
选择，完全相同选择较小 clip/LR。搜索完成后冻结 selected_configs.json，
再运行 seeds 20261011/12/13 下全部六个 cell，共 18 个 final trials。

四个 worker 分别使用 GPU 0/1/2/3，每 trial 单 GPU，无 DDP。GPU 完成后
立即取当前阶段下一 trial，stdout/stderr 写该 trial 的 train.log。
任一 trial 返回非零，则当前队列结束后 pipeline 非零退出，下一阶段不启动。
已完成的 trial 按配置和必要产物验证后复用，绝不重新写入其目录；不完整目录
直接失败。最终汇总文件可由已完成 trial 重新生成。shell 使用调用者的 curve 环境。

配对 RNG 使用相同的 seed、初始模型、固定 permutation 和独立的 seeded
train DataLoader generator；workers 使用 PyTorch 的 worker seed 规则。
噪声使用独立 generator(seed+1)，test loader 使用独立 generator(seed+2)。
每 epoch 还记录首个 physical batch 前四张实际增强图像的 SHA256，用于
检查相同 seed 的增强指纹；最终配对检查也比较 classifier、模型、checkpoint、
训练顺序、坐标和 MF factorization。

机制分析使用 flattened trainable parameter 顺序，NumPy default_rng(0)
不放回采样并排序 2048 个坐标；全 cell/seed 共用它们。Adam.step 后读取
当前 completed vhat，计算 `p=1/(sqrt(vhat_current)+adam_eps)`、`r=p/s`；
`s` 是本 step 实际使用的 previous-state Scale，Standard 为 ones。
`mechanism_trace.npz` 保存 coordinate_indices、r_trace、s_trace。
`mechanism_metrics.csv` 的分位数、mean/std、CV、anisotropy 和 drift 均基于
这 2048 个坐标；coordinate std 为 population std，首步 drift 留空。
这些是 linearized effective noise multipliers，不是 Adam 精确 Jacobian。

MF distortion 读取各 final trial 实际保存的 `M`、对应 `W` 和完整 r_trace，
使用 `H=(W.T@W)*(M@M.T)` 的指定公式；每 trial 2048 坐标的 ratio
p10/p50/p90、logabs mean/median 写入 mf_distortion.csv。
mechanism_summary.json 的 step 指标先对每 trial 求均值，再等权对三个 seed
求均值；drift 不含第一步。MF ratio median 为三个 trial median 的 median，
logabs mean 为三个 trial coordinate mean 的平均。它是线性化 diagnostic，
不是精确 Adam noise variance。结果不自动作因果结论。

final_multiseed.csv / final_summary.json 保存 accuracy（fraction）及 seed sample
std（ddof=1）。factorial_effects.csv / json 保存每 seed 的 G_iid、G_prefix、
G_momentum、I_prefix、I_momentum，及其跨 seed mean/sample_std。
mechanism_summary.json 显式列出三个 Standard/Scale pair。
所有新代码、临时文件、测试产物、日志和 checkpoints 位于 exp2/。

从仓库根目录启动完整 pipeline 的唯一命令：

```bash
conda run --no-capture-output -n curve bash exp2/run_sweep.sh
```

正式实验前验证：

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp2/cache/tmp" conda run --no-capture-output -n curve python -m pytest -c exp2/pytest.ini exp2/tests --basetemp=exp2/results/pytest_tmp
conda run --no-capture-output -n curve bash exp2/run_sweep.sh --smoke
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n curve python -m exp2.verify_smoke
```

smoke 每 cell 只训练一个完整 logical batch、测试 100 张图，依然按完整
250-step strategy 校准。smoke 不参与搜索、冻结配置或最终汇总。
