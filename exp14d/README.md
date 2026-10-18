# Exp14d

从仓库根目录运行，所有写入限制在本目录；历史包不导入其 initializer。

```
conda run --no-capture-output -n curve python -B -m exp14d.tests
conda run --no-capture-output -n curve python -B -m exp14d.pipeline --smoke
conda run --no-capture-output -n curve python -B -m exp14d.start_background
```

后台顺序：A 八次搜索 → 与 Exp14c 全网格合并冻结 → 变化配置的十种子正式训练 → B 两任务各十八次搜索 → 十二组配置冻结 → 120 次正式训练 → 报告和历史只读审计。每卡最多一个训练进程，繁忙卡等待，不自动重试；失败后本批其他任务完成，流水线停止并记录失败，不进入下一阶段。

方法 ID 的 `-adam3` 后缀区分 AdaM³ 与 Adam。矩阵读取 Exp9 同任务/阶段的已验证矩阵，每次按当前 C 重新校准。数据、初始化和随机流复用 Exp14c/Exp9；NLP 正式训练沿用 62,000 条历史训练子集以保持 310 步。

`results/pipeline_status.json` 查看阶段，`scheduler.jsonl` 和各阶段 `*_status.json` 记录 PID、完整命令、GPU 和日志。每个 trial 下保存配置、指标、噪声矩阵、训练顺序、检查点及哈希。`results/final_report.md` 只在完整流水线完成时生成。

Scale C=1 在缩放梯度空间定义，不等同欧氏 C=1。ε=8 仅单次训练预算，无搜索/多次运行组合隐私声明。历史官方评价集已被使用，结果不是新盲测。
