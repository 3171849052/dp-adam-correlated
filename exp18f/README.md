# Exp18f

独立、离线的 STP-Ex2 超参数转折点与整齐参数复核实验。所有代码、运行缓存、日志、检查点和报告写入本目录；只读复用 `exp18e/assets` 已有分词结果和划分、`data/sst2` 原始数据及 `cache/bert-tiny` 预训练模型。

- 第一阶段：13 学习率 + 6 clip + 8 优化器 eps，共27个轴条目；每轴保留基准，108次训练。
- 第二阶段：最多3轮几何中点搜索；若非单调则对标签变化最大的局部间隔加密，最多18次训练。未观测到边界时明确报告删失。
- 第三阶段：0.025、0.03、0.035，clip=100、eps=0.001；五个全新配对种子，共30次训练。只依据这批种子的结果冻结共同参数。
- 两种方法均从缓存重新初始化，固定每种子的训练顺序、dropout种子和噪声种子；无官方验证集选择、无DDP。
- 每任务重新计算敏感度及噪声，单次训练epsilon=8、delta=1e-5，无抽样放大；搜索和复核的组合不宣称epsilon=8。
- 数值失稳记为`numerical_failure`并继续；基础设施故障停止当前阶段，无静默重试。

验收顺序（均在仓库根目录）：

```bash
conda run --no-capture-output -n curve python -B -m exp18f.setup
conda run --no-capture-output -n curve python -B -m exp18f.test
conda run --no-capture-output -n curve python -B -m exp18f.scheduler smoke
conda run --no-capture-output -n curve python -B -m exp18f.accept
conda run --no-capture-output -n curve python -B -m exp18f.start_background
```

`start_background`通过独立会话和重定向标准流脱离终端，等效于nohup，写入`launch.json`、`logs/launcher.pid`及`logs/pipeline.log`。控制器的实际Python PID保存在`logs/pipeline_pid.json`和`pipeline_status.json`。禁止重复启动；不自动重试不完整目录。

查看进度：

```bash
conda run --no-capture-output -n curve python -B -m exp18f.status
```

`search/records.csv`及各轴CSV在每个任务结束时刷新；五种子统计、配对差、搜索最优对照及最终曲线在流水线结束时自动生成。`rounded_freeze.json`在复核完成前明确标记未冻结，`turning_point.json`在学习率搜索中更新；`final_report.md`持续更新，不能把运行中的占位状态当作完成结果。

验收对代码、初始计划和输入文件做SHA256封存；每个任务保存配置、模型初始化/排列哈希、隐私校准、逐步日志、逐轮指标、终态和产物校验和；完成时再核对全部产物及Exp18e目录未改变。
