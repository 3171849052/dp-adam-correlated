# Exp15b

六组优化器/噪声实验：`momentum`、`momentum+iid`、`momentum-bias`、`stpex`、`stpex2`、`stpex-adam3`。新增实现与产物仅写入本目录，Exp9/14/15、数据和预训练模型只读。

仅使用物理 GPU **0、1、2**，GPU **3 留空**。调度器每卡最多一个训练进程，外部占卡时等待。训练入口也拒绝 GPU 3。源环境为 `/home/longt29/miniconda3/envs/curve/bin/python`。

完整顺序：历史审计 → 单元测试 → CV/NLP 两步 smoke 与历史逐张量数值等价验证 → 冻结网格 → 扩展搜索 → 三种子复核 → 正式十种子 → 审计和报告。

```bash
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp15b.audit
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp15b.checks
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp15b.smoke
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp15b.start_background
```

实际启动前会生成 `results/test_results.json`（测试日志哈希和通过数）；流水线要求算法核心文件与验收时一致。`results/phase1_acceptance.json` 是 smoke 验收凭据。完整训练无须额外确认。

扩展网格 CV 60、NLP 90；与旧搜索重合 16 个，新增搜索 134 次，与旧 90 个合并成 224 个候选。方案 A 固定 C/R=1，是方案 B 的子集。按最后一轮内部验证准确率、损失、trial ID 排名前二，再增加种子 20261102/20261103。最终按三种子平均验证准确率、平均损失、trial ID 选参。两方案共享的搜索、复核、正式配置均去重。正式最多 240 次，复核最多 96 次。

查看 `results/pipeline_status.json`、`results/pipeline.log`、`results/scheduler.jsonl` 以及各阶段 `*_status.json`，可找到当前阶段、PID、GPU 和每个 trial 的日志。成功且通过哈希检查的结果不会重跑。部分运行、OOM、校准、公式及来源异常停止阶段，不自动重试；搜索/复核非有限值保留失败记录并排除。正式训练非有限值停止。修复后只能显式恢复，不能直接覆盖旧产物。

历史结果通过小型 `summary.json` 引用复用，`reuse.source_directory` 指向只读原文件，避免复制检查点。所有历史训练已检查完整协议、核心代码、原始文件、配置、噪声矩阵及隐私校准哈希；只在本地五方法 smoke 与历史逐张量完全一致后启用复用。IID 不复用 Exp9 Adam。

最终 `results/final_report.md` 仅在所有必要阶段通过后生成。包含 `paired_differences.csv`、`controlled_differences.csv`、`noise_comparison.csv`、`historical_comparison.csv` 及完整协方差文件。固定超参数比较使用匹配搜索/复核配置及实际可用的正式配置，分别标明内部验证和官方评价范围；十种子调优后配对比较覆盖所有六组。不能把单种子受控验证差称作十种子官方机制效应。

单次训练 ε=8、δ=10⁻⁵；无搜索/重复训练组合隐私声明。官方评价集已使用过，不是新盲测。协议详见 [protocol.md](protocol.md)。
