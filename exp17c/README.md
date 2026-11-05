# Exp17c
独立 r=0 DP-Adam-LC α 搜索，物理 GPU 1、2、3，每卡一进程。

为隔离历史实验模块导入时的缓存路径副作用，数据、模型、裁剪、机制和核算模块逐字复用 Exp17b 实现，仅将包名替换为 exp17c；未修改历史文件。训练沿用其 Poisson 采样、FP32 Adam 状态和完整随机状态检查点。曲率用 FP64 解析计算，更新方向转回参数 dtype；不平滑、不裁剪曲率。

运行环境：/home/longt29/miniconda3/envs/curve/bin/python。
首次验收：python -B -m exp17c.prepare
后台启动/中断后恢复：python -B -m exp17c.start_background
刷新报告：python -B -m exp17c.report

results/background.json 保存调度器 PID 和日志路径。
results/pipeline_status.json 保存阶段、任务 PID、GPU 分配与完成数。
results/trials/*/{config.json,progress.json,diagnostics.jsonl,evaluations.json,checkpoint.pt,summary.json,worker.log} 保存逐实验记录。
每 10 步和末步写原子检查点，恢复模型、Adam、采样、噪声和全部 RNG 状态；重跑已完成任务时校验哈希后跳过。异常任务保留失败记录并停止阶段，不自动掩盖数值失败；普通中断无失败摘要时可从检查点恢复。
每步记录验证（D 为冻结后的测试）交叉熵和准确率；报告在阶段结束时自动更新，也可手动刷新。
首次步的 input_min_abs 是所有参数坐标的最小绝对带噪输入，input_zero_count 是恰好为零的坐标数。范数及分位数覆盖全部坐标，分位数取修正绝对值，余弦取被减修正与基础方向。
边界扩展最多一轮：B 非零最优位于初始边界且胜过基线时，补测相邻数量级两个点；剩余边界不确定性写入 final_freeze.json，不声称找到全局最优。
每次 ε 独立核算，多次调参不具有联合 ε=8 保证。95% CI 为配对 t 区间，未经多重比较校正。

报告表格依赖 tabulate，已隔离安装到 runtime/vendor。实际后台启动/恢复请运行 bash exp17c/start.sh（自动传递该依赖目录）；单独刷新报告时设置 PYTHONPATH="$PWD/exp17c/runtime/vendor"。
