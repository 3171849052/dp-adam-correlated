# 整齐数值的学习率继续搜索

原 Exp18/Exp18b/Exp18c 结果及 data/ 保持只读。仅本目录存放新增代码、冻结协议、逐轮计划、日志、检查点及结果。

首轮：Adam 0.08/0.10/0.15；Momentum 1.0/1.5/2.0；STP-Ex2 0.10/0.15/0.20。
后续每轮只扩展未出现转折的优化器，从 1、1.5、2、3、4、5、6、8 乘以 10 的整数次幂的序列中，取高于当前最大 LR 的三个整齐数值。

每种优化器的 IID、B2LT、两者等权平均三条五种子验证曲线均需满足：最高两个 LR 都高于该视角最优 LR 且准确率更低，最高 LR 至少下降 0.1 个百分点。没有预设最大轮数。数值发散保留现场、记录为边界，不虚构验证分数；基础设施错误停止派发。该规则属于观测转折，不代表全局最优或显著性检验。

除 LR 外所有参数固定；直接复用原 CNN、优化器、噪声、隐私和统计函数。`train.py` 沿用 Exp18c 训练循环，新增按轮次读取计划及数值发散记录。每轮先冻结 `rounds/NNN/plan.json`，完成最高 LR 的 IID/B2LT 冒烟再正式运行，每个 LR 10 次完整训练。已完成优化器不会追加候选。

搜索及停止均仅依据验证集。全部转折后在 Exp18b/Exp18c/本次新候选的合并网格中选择每种优化器的共同 LR，冻结 `best_hyperparameters.json` 后再评价 30 个检查点；同一检查点已有官方测试结果则按哈希复用，不重复测试。

```bash
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp18c.continuation.accept > exp18c/continuation/acceptance.log 2>&1
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp18c.continuation.start_background
```

使用四张自动检测并按 UUID 绑定的 GPU；显存动态准入，上限每卡 12 独立进程。后台使用独立会话、stdin=/dev/null、文件锁；PID/命令见 background.json，状态见 status.json，主日志 pipeline.log，每任务日志 trials/<id>/worker.log，遥测 gpu_telemetry.csv。

`trial_plan.csv` 为累积新训练计划，`grid_per_seed.csv`/`grid_summary.csv` 为合并网格，`new_grid_per_seed.csv` 仅本次训练，`turnover.json` 为九条曲线的停止证据。完成后生成正式统计、配对差异和 final_report.md。协议、代码、矩阵、初始化在首次验收后冻结，每轮计划另行冻结；扩展规则本身不变。
