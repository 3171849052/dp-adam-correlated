# Exp16a

14 方法、两任务、完整五轮训练。标准 Adam 与 IID，三种 workload × 三种 MF，以及三种 SGD-MF STP。STP 完全使用 Exp9 Scale：上一轮已完成 vhat 冻结缩放 → 该空间逐样本裁剪和加噪 → 撤销缩放 → 标准 Adam；不是 STP-EX。所有 Adam β=(0.9,0.999)、eps=1e-8、weight_decay=0。

在仓库根目录用现有 `curve` 环境执行：

```bash
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp16a.stage1
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp16a.start_background
```

第二条必须在阶段一通过后执行，创建独立会话、断开 stdin、重定向文件日志，以 flock 防止两个调度器同时运行。不需要人工确认。完整流程自动执行粗搜索 → 一轮细搜索 → 前二配置追加两种子复核 → 冻结 28 套配置 → 十种子 280 次正式训练 → 审计和报告。单次入口为 `CUDA_VISIBLE_DEVICES=0 <curve-python> -B -m exp16a.train --trial-json '<完整 Trial JSON>'`；正式/搜索任务必须属于已冻结网格。

进度与日志：

```bash
cat exp16a/results/background.json
cat exp16a/results/pipeline_status.json
tail -f exp16a/results/pipeline.log
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp16a.audit
```

`*_status.json` 保存任务计划、状态、PID、物理 GPU、命令及日志位置。`scheduler.jsonl` 为追加事件日志；`gpu_telemetry.jsonl` 记录占用。每张物理卡 0/1/2/3 最多一个 CV，或两个独立 NLP，二者不共存；worker 使用 CUDA_VISIBLE_DEVICES 并在本进程用 cuda:0，不使用 DDP。外部占卡等待。OOM 停止新任务，不改批次或 FP32。

阶段一：本地资产哈希与只读快照 → 30 组矩阵（每时域 9 个 MF+1 IID） → 数学、梯度裁剪、模型等价和工作流测试 → 46 次真实两步 smoke（28 任务/方法、14 CV 正式时域、4 复现）。Smoke 只评价 64 条内部验证样本，不替代完整训练。通过后 `phase1_acceptance.json` 冻结核心、平台代码、矩阵及验收证据 SHA256。

矩阵只依赖公开 workload/训练长度/固定参与协议，各组合求解一次。SGD 普通 MF 与 STP 引用同一文件。所有 A 保留任务书原始定义，无额外全局常数。BandInvMF 使用 4 带逆 Toeplitz、论文 BISR 起点及 L-BFGS；Momentum-Bias 使用完整非 Toeplitz A。B²LT 按随附 PDF 第 5 节使用 P=K=T/5、Q=5、有限指数混合和全部障碍项；按第 6 节 Q 缓冲与种子重放超出带宽的旧创新。DTC 直接优化全部单调非负 Toeplitz C 系数，按第 7 节初始化、障碍和 IID 候选选择，完整创新历史存放在主机 FP32 RAM，不占 T 份 GPU 存储。

所有相关机制使用同一固定五轮参与的绝对 Gram 敏感度上界、add/remove zero-out、无采样放大 GDP。由目标 ε=8、δ=1e-5 解 μ，再令 σ_sum=clip·Sens(C)/μ；不使用 PDF 的另一套保守校准公式。单位裁剪协方差、逆关系、完整 A 目标、敏感度上下界、实际 ε、在线流及经验协方差均独立检查。矩阵指标用于设计线性 workload，不能声称等同于自适应 Adam 参数轨迹误差。

搜索种子 20261101，348 次粗训练；每方法自己的最优点按 log 半区间生成 3 个候选/维，边界最优向外半区间扩展，保留最优点。细网格总预算仍为 348，成功的同配置中心点复用，正常情况下新增 320 次。合并后每方法前二用 20261102/3 复核（112 次），按三种子平均内部准确率、平均验证损失及 ID 选优。正式 20261111–20261120，共 280 次。全部成功时共 1060 次不同完整训练；失败的搜索/复核非有限候选保留并排除，不重试。完整预注册参数见 `protocol.json`、`results/protocol_freeze.json` 和各 grid。

恢复时重新执行后台入口，只复用身份、配置、核心、输入、矩阵和全部产物哈希一致的成功任务。未完成/工程失败现场要求明确检查，不自动覆盖或重试；没有自动续训部分 checkpoint。搜索/复核明确记录的 non_finite 可排除；其他失败停止，正式任何失败停止。所有失败保留 summary、traceback、日志和已写指标。

每个 trial 保存配置、训练顺序、逐步损失/裁剪/梯度范数/预算/时间、逐 epoch 内部或官方指标、模型和 Adam checkpoint、内存峰值及哈希。最终 `summary.csv`、`utility.csv`、`paired_differences.csv`、`matrix_errors.csv`、`sensitivity.csv`、`runtime_memory.csv`、`final_audit.json` 和 `final_report.md` 只在实际所需阶段完成后生成。CI 为十个种子的 Student-t 95%，差异按相同种子配对，未做多重比较校正。

历史实验、data/、cache/ 仅本地读取，未导入历史 package initializer。沿用 Exp9 训练/验证划分、模型、预处理、随机初始化、数据顺序、CV 增强与 NLP dropout；所有缓存及临时文件均在 exp16a/runtime。参考来源和历史结果差异见 `historical_review.md`。官方测试只在 28 套配置冻结后读取；历史已经使用过官方评价集，本次不声称新盲测。ε=8 是单次训练预算，内部验证假定公开，调参/复核/重复训练不共享一个 ε=8 保证，原始梯度诊断是私有审计产物。
