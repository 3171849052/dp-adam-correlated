# Exp13：固定创新、实际噪声平方驱动的自洽 Workload

本目录只进行相关噪声的数值模拟与四带矩阵优化。**不构建或执行模型，不做训练、
反向传播或模型参数更新**。只读 `data/` 和 `cache/`，不下载资源、不导入历史实验包。
所有新增源码、缓存、测试证据、矩阵、日志和结果均在 `exp13/` 内。

## 数学定义

所有计算使用 Float64。令 `Z` 为下移矩阵，

\[
D(d)=I+d_1Z+d_2Z^2+d_3Z^3,\qquad d=(1,d_1,d_2,d_3)^\top,\quad C=D^{-1}.
\]

沿用 Exp9 的固定五次参与模式：`spacing=T/5`，参与列为
`j, j+spacing, ..., j+4*spacing`。逆矩阵敏感度使用相同的绝对 Gram 上界，
允许不同参与时的梯度方向不同：

\[
S(C)^2=\max_{0\le j<T/5}\sum_{p,q=0}^{4}
  |(C^\top C)_{j+pT/5,\;j+qT/5}|.
\]

GDP 标定为 `epsilon=8, delta=1e-5`，add/remove zero-out，无抽样放大。
求解 `delta(mu,epsilon)=1e-5`。采用共同单位裁剪范数 `C_clip=1`、Exp9 的逻辑
batch `B=1000`，在平均梯度噪声空间中

\[
\sigma(d)=\frac{C_{clip}S(C)}{B\mu},\qquad N(d)=\sigma(d)D(d)\Xi.
\]

这里 `C_clip=1` 是数值模拟的共同单位标定，不读取某次训练选出的裁剪参数。
学习率取 1，因此两种误差均为单位学习率下的累计误差；共同学习率会将误差乘以其平方。
`sigma` 是创新标准差，逐时刻边际标准差是
`sigma*sqrt(sum_{l=0}^{min(t,3)} d_l²)`，报告中分别给出。

每个拟合种子一次性用 NumPy PCG64 生成 `Xi[T,K]`，再复制到分配的 GPU。
**同一次拟合永不重新采样、永不使用其他种子的创新参与优化。**
从零初态递推，`beta1=.9, beta2=.999, eps=1e-8`：

\[
v_{t,c}=\beta_2 v_{t-1,c}+(1-\beta_2)N_{t,c}^2,\quad
\hat v_{t,c}=v_{t,c}/(1-\beta_2^t),\quad
A_{t,c}=a_{t,c}=\frac1{\sqrt{\hat v_{t,c}}+10^{-8}}.
\]

实现通过下三角矩阵 `B2[t,s]=(1-beta2)*beta2^(t-s)/(1-beta2^t)` 乘以实际
`N²` 计算上述递推；与顺序递推在 Float64 舍入误差内相同，单元测试覆盖三个正式 T。
**实际自洽方法不使用 `E[N²]`。**

用包含完整一阶动量偏差修正及 `1-beta1` 的矩阵

\[
M_{t,s}=\mathbf1_{s\le t}\frac{(1-\beta_1)\beta_1^{t-s}}{1-\beta_1^t},
\quad L_{t,s}=\mathbf1_{s\le t},\quad W_c=L\operatorname{diag}(a_{:,c})M
\]

定义 Workload 的 Frobenius 目标：

\[
J(d)=\frac{\sigma(d)^2}{TK}\sum_{c=1}^K\|W_c(d)D(d)\|_F^2
    =\sigma(d)^2 d^\top Q(d)d.
\]

坐标无需近似、采样或逐坐标保存 `T×T` 矩阵。令 `H=LᵀL`，先精确合并：

\[
G=\sum_c W_c^\top W_c
 =M^\top[H\odot(AA^\top)]M,
\qquad
Q_{i,j}=\frac1{TK}\sum_{r=0}^{T-1-\max(i,j)}G_{r+i,r+j},\quad 0\le i,j<4.
\]

**每次候选目标评价均重新计算 `sigma → N → vhat → A → G → Q → J`，没有固定 Q 的外层迭代。**
Gram 公式保留了分母在不同时刻之间的坐标相关性，不能用平均 `a` 替代 `AAᵀ`。

另算完整纯噪声 Adam：`mhat=M N`，使用同一实际噪声平方得到的分母，

\[
U_{t,c}=\sum_{s\le t}\hat m_{s,c}a_{s,c},\qquad
J_{real}=\frac1{TK}\sum_{t,c}U_{t,c}^2.
\]

负的更新符号不影响平方误差。这里 `U` 只是模拟累计更新数值，没有任何模型参数被更新。
两种误差都平均所有前缀和坐标，**不是只计算最终前缀**。
`J` 是带内生分母的 Frobenius 代理目标，`J_real` 是该创新样本上实际累计误差。
由于 `a` 本身依赖 `Xi`，不能将 `J` 声称为 `J_real` 的无偏估计，优化 `J`
也不保证 `J_real` 同步改善。

## 比较方法与优化设置

1. `iid`：`d=[1,0,0,0]`，使用相同的固定五次参与标定及实际噪声评价。
2. `exp9_momentum_bias`：复现 Exp9 原始固定 Momentum-Bias Workload 的优化。
   为保持其优化器数值尺度，拟合时仍省略全局 `1-beta1`；该常数不影响最优系数。
   最终比较的 `J`、`J_real` 均使用上面的完整 Adam 定义。
3. `expected_self_consistent`：每次候选重新计算 `sigma` 和边界随时间变化的
   `E[N_t²]=sigma²*sum_{l<=min(t,3)}d_l²`，以其递推 `vhat`、重算 Q 并优化。
   这是唯一使用方差期望代替实际平方的比较项；同样在每个种子任务内重新拟合。
4. `actual_self_consistent`：每个种子独立最小化实际 `N²` 定义的 `J(d)`。

两种自洽方法以及原始 Exp9 使用相同的 Nelder–Mead 设置：
`maxiter=1600, xatol=1e-9, fatol=1e-7`，两个固定起点分别为 Exp9 的四带
cumulative-momentum BISR 系数和 IID，`d0=1`。BISR 初值在本目录用相同的
幂级数平方根/逆递推生成，无需加载 JAX 或历史 Python 包。

保留 Exp9 的数值搜索域 **`max(abs(D^-1)) <= 1e8`**；越界候选优化器取 `1e100`。
自洽方法即使遇到越界候选，也先完整重算实际目标，并在轨迹同时记录原始 `J`、
`optimizer_value` 和 `admissible`。这一约束需要明确保留：自适应分母会抵消大量
噪声尺度效应，无约束搜索可能向极大逆放大方向移动。结果给出逆矩阵最大元素、
谱范数和 Frobenius 范数，便于判断解是否接近边界。
数值异常或最优起点未收敛会直接报错，无重试、回退、静默异常处理或自动修复。
原始 Exp9 的种子无关基线允许在同一工作进程内按 T 缓存；不缓存或复用任何
实际/期望自洽拟合结果。

## 任务配置和设备

正式设置：`T∈{225,250,310}`，`K=4096`，十个独立拟合种子 `20261111–20261120`。
对应 CV 搜索 45000、CV 正式 50000、NLP 62000 个样本，五个 epoch、batch 1000。
本实验不训练，T 仅决定模拟长度和参与间隔。

`assets.py` 只读 CIFAR-100 训练集、SST-2 训练 parquet、缓存 ViT safetensors 及
BERT 配置/权重，确认任务规模和参数结构，并记录 SHA256。ViT 将 checkpoint 的骨干
形状映射到 Exp9 的 ViT-Tiny 结构，替换为 100 类 head；BERT 排除 MLM head 和非浮点
buffer，并加入二分类 head 的形状。无需构造模型或生成分类 head 的随机数值。
按参数名排序后的展平参数结构等距选出 K 个不同坐标，每个模拟列对应一个真实参数坐标。
每列在目标中权重相同，不代表层级加权或整个模型的全部参数。
数据内容、标签和模型参数数值不参与噪声优化，未读取官方测试/验证集。

GPU 使用 `0,1,2,3`，每张卡一个独立进程；同一种子的三个 T 都在一张卡上完成。
种子按列表下标轮转分配，任务清单写入 manifest。子进程通过 `CUDA_VISIBLE_DEVICES`
隔离设备并使用本地 `cuda:0`。开启确定性运算，禁用 TF32，固定 cuBLAS 配置；
创新由 CPU 生成，因此不随 GPU 编号变化。同一软硬件版本可复现，跨版本不承诺逐位相同。

每个拟合矩阵额外在三个独立新种子上评价，所有方法配对使用相同的评估创新：
`test_seed = fit_seed + 100000000 + i*1000000`，`i=0,1,2`。
评估种子与拟合种子不相交，且不在不同拟合种子之间复用；它们不用于选择矩阵或起点。
期望二阶矩比较项的拟合本身不依赖 Xi，因此各拟合种子的系数通常相同。

## 运行

从仓库根目录执行，使用 Conda `curve`。先运行单元测试：

```bash
conda run --no-capture-output -n curve python -B -m pytest exp13/tests -q -o cache_dir=exp13/runtime/pytest_cache --basetemp=exp13/runtime/pytest_tmp --junitxml=exp13/results/unit-tests.xml
```

最小四卡 smoke：4 个拟合种子，每卡 1 个，`T=10, K=16`，完整优化四种方法，
每个矩阵用三个新种子评价，验证资产读取、日志、矩阵、聚合报告整条路径：

```bash
conda run --no-capture-output -n curve python -B -m exp13.run --smoke --gpus 0 1 2 3
conda run --no-capture-output -n curve python -B -m exp13.verify_smoke exp13/results/smoke
```

只查看正式分配和协议，不启动优化、不读取模型资产：

```bash
conda run --no-capture-output -n curve python -B -m exp13.run --gpus 0 1 2 3 --plan
```

**正式完整数值实验（本实现阶段没有启动）：**

```bash
conda run --no-capture-output -n curve python -B -m exp13.run --gpus 0 1 2 3 --coordinates 4096 --holdout-seeds 3 --output exp13/results/full
```

命令会执行 30 个种子×T 任务；各任务单独运行两个自洽优化，每个优化两个起点。
默认不从旧结果续跑，输出目录已存在即报错。重复 smoke 或正式运行需显式选择新的
`--output exp13/results/<new-name>`，保留既有证据。`--smoke` 结果不得当成正式结论。
启动前应检查 GPU 可用资源；脚本不获取 GPU 锁、不自动改设备或降规模。

可单独从已完成结果重建报告：

```bash
conda run --no-capture-output -n curve python -B -m exp13.report exp13/results/full
```

## 结果格式

`results/<run>/` 下：

- `manifest.json`：固定协议、逐 GPU 任务分配、源码/资产哈希、库版本和启动命令。
- `assets.json`：只读输入哈希、数据规模、完整参数形状和 K 个模拟坐标映射。
- `gpu{0..3}.log`：各进程启动、优化完成、耗时和错误栈。
- `T{T}/seed{seed}/xi.npy`：拟合时唯一固定的 Float64 创新矩阵。
- `.../{expected,actual}_self_consistent_trajectory.jsonl`：每次优化器目标评价，
  包含系数、重算的 J、优化器值、是否在 Exp9 数值域内和模式。
- `.../{method}_matrices.npz`：`d[4]`、`D[T,T]`、`C[T,T]`、在该种子实际 `N²` 下
  重新计算的 `Q[4,4]` 和 `sigma`。期望模式拟合目标另外存入 summary 的 fits。
- `.../summary.json`：每个起点的收敛状态、迭代/评价数、拟合目标、系数、拟合和留出
  的 `J/J_real`、相对 IID/Exp9 的改善百分比、创新/边际噪声标准差、逆放大、
  `vhat` 全部/末步分布、各时间步的七个分位数、创新与矩阵哈希及耗时。
  分位数顺序为 min、p01、p05、p50、p95、p99、max；vhat 标准差为总体标准差。
  `in_sample[method].J_expected` 另记录该矩阵的期望分母 Workload 目标。
- `metrics.csv`：每个拟合/评估种子×方法一行，原始指标、系数和改善百分比。
- `statistics.json`：各 T、方法、拟合/留出上的均值、样本标准差、Student-t 95% CI、
  最小/最大值；包括系数、误差、改善、噪声、逆放大和二阶矩分布。
- `paired_differences.json`：实际自洽相对三个比较项的配对差值统计（actual−baseline）。
- `overfit_gaps.json`：拟合创新上的改善百分比减去新创新上的平均改善，正值表示
  改善在新创新上减小。误差有 J/J_real 两种，基线有 IID/Exp9 两种。
- `report.md`：简要汇总，明确区分 J 与 J_real。
- `completion.json`：所有任务及报告成功完成后才生成。

改善定义为 `100*(1-error_method/error_baseline)`，正值表示误差下降。
跨种子先计算配对百分比，再求均值，不能用两个总体均值之比冒充。
留出统计先在同一拟合种子内平均三个评估结果，再在十个独立拟合种子间求标准差和 CI，
避免把同一拟合矩阵的三次评价当成三个独立拟合样本。报告不预设任何方法一定改善。

测试覆盖：完整偏差修正、逆矩阵/参与敏感度、GDP、显式逐坐标 Workload 与 Gram/Q
等价性、实际 Adam 顺序递推、期望分母边界、候选间重算/固定创新、重复拟合复现性、
正式三个 T 的少坐标前向计算、四卡 Float64 一致性、种子/坐标映射和统计。
