# Exp18g — STP-B²LT

仅5次正式训练，无搜索。固定 `lr=0.01, C=500, eps_scale=0.001`；Adam `betas=(0.9,0.999), eps=1e-8, weight_decay=0`。

逐样本缩放裁剪及Adam逻辑批次直接移植Exp16a（与Exp16e的Adam路径相同）：在上一已完成步的偏置修正Adam二阶矩上冻结 `s=1/(sqrt(vhat)+eps_scale)`，裁剪`s*g_i`，加入Exp18d SGD-B²LT相关噪声，除以`s`并除以1000，再执行Adam。没有自适应α。

只读复用本地模型、数据、已验证分词与划分。Exp18f正式协议：FP32、关闭TF32、长度128、物理/逻辑批次1000、累积1、全参数微调。训练62000条，内部验证5349条仅监控。五轮310步后保存最终模型，再对872条官方验证评价一次。初始化、数据排列及逐步Dropout种子与Exp18f配对。

矩阵直接读取 `exp18d/matrices/sgd.{json,npz}`，不使用旧Exp16a矩阵。矩阵参数、哈希、逆关系、SGD工作负载、求解器终点收敛与一阶审计均保存到 `acceptance/matrix_audit.json`。Exp18d保存的最低未惩罚误差矩阵与障碍目标终点按原协议区分。

每次训练单独校准 ε=8、δ=1e-5；无抽样放大；五次组合不宣称 ε=8。内部诊断不是DP发布。

在仓库根目录依次运行：

```bash
conda run --no-capture-output -n curve python -B -m exp18g.prepare
conda run --no-capture-output -n curve python -B -m exp18g.test
conda run --no-capture-output -n curve python -B -m exp18g.scheduler smoke
conda run --no-capture-output -n curve python -B -m exp18g.accept
conda run --no-capture-output -n curve python -B -m exp18g.start_background
```

四张GPU按UUID隔离，各最多一个进程；四任务完成后自动调度第五个。启动器通过nohup、独立会话和日志重定向保持后台运行，`launch.json`保存启动PID及命令，`logs/pipeline.pid`保存启动器PID，`pipeline_status.json`保存实际控制器PID，`logs/formal_events.jsonl`保存工作PID及命令。失败会停止阶段并保留日志，不自动重试或重复训练。

完成后自动生成 `results/per_seed.csv`、`summary.csv`、`seven_algorithm_summary.csv`、`paired_statistics.json`、`final_report.md`。前六算法直接复用 `exp18f/final_official/per_seed.csv`，五种子逐一配对，SD使用ddof=1，双侧95%CI使用Student-t(df=4)。内部监控单独输出。

进度：`conda run --no-capture-output -n curve python -B -m exp18g.status`。
