# Exp18e: SST-2 / BERT-Tiny paired private fine-tuning

所有派生文件均在本目录；只读取仓库已有 `data/sst2/` 与 `cache/bert-tiny/`，禁用联网。所有 Python 进程使用 Conda `curve`。

- `protocol.json`：完整协议（沿用 Exp18d 的数据划分、算法、FP32、310步及固定参与隐私会计）。
- `search_space.json`：粗网格、几何中点邻域、边界扩展、三种子复核与共同参数选择规则。
- `plans/`：每个阶段在执行前落盘的完整任务清单。`formal_template.csv` 有30行；冻结后生成数值完整的 `trial_plan.csv` 与 `plans/formal.json`。
- `matrices/`：从 Exp18d 复制的矩阵和原审计；`matrix_audit.json` 保存重审计；`upstream_manifest.json` 保存只读源的哈希。
- `search/*_calibration.json`：每组候选、两种机制的噪声校准；每个工作进程还会独立计算并保存 `privacy.json`。
- `search/records.csv` 与 `search/*_ranking.json`：全部搜索训练记录与配对验证集排序。粗搜索121组/242次训练；精搜索不超过14组/28次；边界不超过2组/4次；复核固定12次。总搜索254–286次。
- `selection_freeze.json`：仅以复核种子20261103/04/05的内部验证均值冻结两种STP共同参数。`freeze_manifest.json` 防止正式阶段参数改变。
- `formal/`：六方法×五种子，共30次全新完整训练；正式种子严格为20261111–20261115。每个任务保存配置、隐私校准、初始化/排列哈希、逐步日志、逐轮内部验证、最终官方验证、模型/优化器检查点、峰值显存与时间。
- `summary.csv`、`per_seed.csv`、`paired_differences.csv`、`paired_statistics.json`、`final_report.md`：冻结后正式阶段逐步生成，全部完成后才标记完成。准确率、损失及配对差均使用样本SD、Student-t 95% CI。
- `logs/gpu_telemetry.jsonl`：每秒四张物理GPU的利用率、显存及活跃训练PID；结合阶段events文件可关联任务。

先执行（已由本次实现过程完成）：

```bash
conda run -n curve python -B -m exp18e.setup
conda run -n curve python -B -m unittest discover -s exp18e/tests -v
conda run -n curve python -B -m exp18e.scheduler smoke
conda run -n curve python -B -m exp18e.accept
```

实际启动：`bash exp18e/start_background.sh`。启动器使用 `start_new_session=True`，关闭标准输入、重定向完整日志，不依赖交互会话。PID、启动命令、运行状态分别位于 `logs/pipeline.pid`、`logs/launch_command.txt`、`logs/pipeline_status.json`。已有启动记录时拒绝重复启动。

监控：`tail -f exp18e/logs/pipeline.log` 或 `conda run -n curve python -B -m exp18e.status`。

动态调度先覆盖四卡，再根据实测空闲显存和冒烟峰值允许每卡第二个独立进程；单进程只可见一个GPU UUID，无DDP。错误会停止同阶段其他任务并保留输出，不会静默跳过或重试。

每次训练为ε=8、δ=1e-5的独立保证，整个搜索及正式实验组合不是联合ε=8保证；内部原始诊断不是DP发布。
