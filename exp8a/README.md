# Exp8a: BERT-Tiny + SST-2

从仓库根目录运行，使用 conda `curve` 和单张 RTX 3080 Ti，默认 GPU 0。
所有代码、测试、tokenized tensors、split、日志、矩阵和结果仅写入 `exp8a/`。
官方 SST-2 **train** 下载至 `data/sst2/`；模型及 vocab/tokenizer 存于 `cache/bert-tiny/`。
训练、评估和 benchmark 仅用本地文件。官方 validation 不下载、不读取。

```bash
conda run --no-capture-output -n curve python -B -m exp8a.stage1 --gpu 0
```

Stage 1 自动下载、生成固定分层 split 和固定排列、统计未截断长度、预先生成
32/64/128 固定 padding 输入，运行所有 CPU/CUDA unit tests，随后 Standard、Scale
各运行一个真实 **1000-example logical step**。Stage 1 不启动 Stage 2。

```bash
conda run --no-capture-output -n curve python -B -m exp8a.stage2 --gpu 0
```

Stage 2 要求当前代码、checkpoint、split 对应通过的 Stage 1。
A: length64 下依次测试 8/20/40/50/100，两种裁剪分别 1 warmup + 3 measured steps。
发生 OOM 加测4；100 比50综合吞吐提升>5%则测试125/200/250。
选择两种裁剪均无异常且 allocator 峰值留20%余量的配置，吞吐用两种机制 steps/s
的 harmonic mean；距最快<5%优先显存更小。
B: 默认最多4个 **62-step (1 epoch)** length64 pilots，选择共用 lr/C/eps_scale；可用
`--pilots 0` 跳过，`--pilots {1,2,3,4}` 限制数量。随后冻结超参数。
每种长度做 benchmark、一个完整310-step Momentum-Bias + Scale run。
任意两长度 accuracy 差距<=0.006 时扩展为3个配对seeds，共9 runs，否则3 runs。
选最短且 paired mean gap 与95%上界均<=0.003 的长度；证据不足优先64。
最后在选定长度重新搜索 physical batch，冻结最终组合。

若128长度 benchmark没有20%余量，先在128重选一个所有长度共用的physical batch；
所有长度的完整训练仍使用相同physical batch。最终physical batch复测可能得到新的计算切分，
logical batch、fixed order、310次optimizer/noise更新及DP预算保持不变。

## 固定协议与数值验证

- `prajjwal1/bert-tiny` 通用预训练 BERT，pooler/CLS + 随机二分类 head，全部参数更新，CE。
- SST-2 train 67,349，split seed20261008；分层62000/5349。固定排列每epoch重用，
  每样本participation为j,j+62,...,j+248，恰好5次；配对seed20261011–13。
- Logical1000，epochs5，steps310；GDP ε8/δ1e-5，add/remove zero-out，无sampling amplification。
- Adam beta1=.9/beta2=.999/eps1e-8/weight_decay0，FP32，eager attention。
- **dropout=0** 是固定训练协议的一部分，避免physical切分改变随机mask；
  初始化用配对seed，噪声独立generator seed+1；TF32关闭，确定性算法开启。
- Standard: Opacus Fast/Ghost两遍backward；Embedding范数使用padding-aware精确sampler。
- Scale: 前一完成step的vhat，s=1/(sqrt(vhat)+eps_scale)，精确逐样本||s*g||；
  两遍backward，累计剪裁raw gradient，再scaled sum空间加一次相关噪声并inverse scale。
- Embedding: 按(sample,token)聚合重复token梯度后再乘scale、取范数，padding导数为0。
  CPU/CUDA测试与逐样本autograd比较范数、clip系数、梯度和physical分割等价性。
  不生成B×vocab×hidden密集张量；其余层沿用Exp7b scaled Fast norm sampler。
- `privacy.py`、`noise.py`、`clipping.py` 核心累积/scale机制来自Exp2/Exp7b验证代码的本地副本；
  `bandinvmf.py`来自Exp7b完整非Toeplitzworkload优化，在310/spacing62重新计算系数与GDP校准。
  不导入Exp7b运行模块，避免其环境初始化写入旧实验。

## 结果与审计

`results/token_lengths.json`, `clipping_correctness.json`, `tests.log`, `tests.xml`,
`stage1.json`, `stage1_gpu_smoke.csv`, `download_manifest.json`, `mechanism.json`。
Stage 1 的搜索CSV只有header；`selected_config.json`状态为`stage1_only`、最终参数null，
**Exp8b必须拒绝这个状态**。只有Stage 2完成后状态为`final`，其中含最终参数、
冻结超参数、split/checkpoint hashes、训练协议、BandInvMF系数/GDP参数和完整选择依据。

`physical_batch_benchmark.csv`, `max_length_benchmark.csv`, `max_length_accuracy.csv`，
`physical_batch_final_benchmark.csv` 和 `exp8a_report.md`在Stage 2记录实测结果。
`runs/`保留每seed/长度的model checkpoint、config、matrices、metrics及summary。
Benchmark在独立进程中运行；CUDA同步计时，记录allocated/reserved峰值、OOM、non-finite、
完整steps/s和samples/s。20%余量按PyTorch allocator峰值计算，驱动context不计入该统计。
每次训练的GDP单次预算ε8，不声称多次公开数据调参的总隐私预算仍为ε8。

```bash
conda run --no-capture-output -n curve python -B -m pytest exp8a/tests -q -o cache_dir=exp8a/runtime/pytest_cache --basetemp=exp8a/runtime/tests
```

可单独运行 `python -B -m exp8a.benchmark_batch --gpu 0` 或
`python -B -m exp8a.train --gpu 0 --physical-batch-size 20 --max-length 64 --geometry scale`。
下载使用官方Hub的固定revision地址和curl（避免本机Python代理TLS兼容问题），
下载revision和SHA256记录在manifest，后续训练不访问Hub。

Embedding 显存大小估算：30522×128 FP32 word embedding 的 dense per-sample gradient
在physical100/250时单张量为约1.455/3.639 GiB，scaled乘法可能再分配同等大小的临时张量。
这些是理论估算；优化后完整logical-step的实际显存见报告。

Stage 2 若在初始 batch 搜索后停止，可显式使用 `--after-batch` 读取已完成的原始
benchmark JSON继续；已完成且协议匹配的pilot计入原有4次上限，不重复训练。
当前pilot序列为 `(lr,C,eps_scale)`：`(5e-4,1,.1)`、`(5e-5,1,.1)`、
`(1e-4,1,.1)`、`(5e-5,3,.1)`；较低学习率的候选根据首个pilot的收敛结果确定。

本轮实际4个pilots和9个full runs未产生有效学习：Accuracy均停在多数类基线55.786%，
validation loss随epoch上升。长度64使用原协议的证据不足回退，不能把相等的基线Accuracy
解释为长度32已经通过非劣比较。最终JSON的`hyperparameters_converged=false`明确标记这一点。
现在pilot阶段要求Accuracy超过多数类基线0.003且loss低于log(2)，否则停止新full runs。
`--after-batch --after-length`只审计已完成的长度runs并复测最终batch，不增加pilot或full run。
最终benchmark使用独立结果目录，保留初始batch搜索的原始worker JSON供审计。

补充容量测试：支持physical batch500/1000；length128下Standard/Scale各1 warmup + 3 measured完整logical steps。1000两种机制均未OOM/non-finite，因此未触发500回退。原始JSON/log在`results/large_batch_128/`，汇总为`results/large_batch_128_summary.json`与`results/physical_batch_1000_500_length128.csv`。
