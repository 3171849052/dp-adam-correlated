# Exp18b

独立的 MNIST/Exp18 CNN 完整网格实验，全部写入 exp18b/，data/ 和 exp18/ 只读。

用户在获知 cache 只有不兼容的 BERT/ViT 权重后，后续指示使用 Exp18 CNN 并后台启动。
因此采用 Exp18 的随机初始化：正式五种子各生成一份 CNN 初始状态，所有方法与学习率按种子共用。
这些文件保存在 initializations/，不是预训练 cache 权重。最初阻塞审计保存在 history/initial_cache_blocker/。

直接复用 exp18.model.CNN、exp18.optimizer.Update、exp18.noise.OnlineNoise、exp18.privacy 和
exp18.scheduler.admission_limit。复制 Exp18 已验收的三个矩阵，核对 Exp18 acceptance 哈希，不重新优化。
算法定义与 stp-ex2.pdf 第三列相同；按要求使用 GDP 替代 PDF 的 RDP 转换。

协议见 protocol.json；90 次完整训练全部为 162 步、batch=1000、FP32。所有超参数与五种子固定。
IID/B2LT 共用初始状态、三轮重复的同一数据排列、相同单位高斯创新流，实际配对哈希逐 trial 记录。
训练只读取训练/验证张量。对两种噪声、五种子最终验证准确率等权平均，为每个优化器选一个 LR，
然后冻结 best_hyperparameters.json。30 个选中检查点各做一次完整官方测试评价，不另行训练。

运行环境是 Conda curve；从仓库根目录运行：

```bash
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp18b.accept
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp18b.start_background
```

验收包含 12 项数学/协议检查和六种方法各 56 步 smoke test，六进程同时在 GPU 1/2/3 各运行两个。
Smoke 沿用第一个正式种子，覆盖 P=54 边界，不计作完整网格训练。验收后冻结代码/协议/矩阵/初始化哈希。

后台启动用 start_new_session=True、stdin=/dev/null、文件日志，等效于脱离终端的 nohup；
启动器把已持有的文件锁传给后台进程，避免重复调度。启动命令、PID、恢复命令在 background.json，
运行状态在 status.json，日志在 pipeline.log，显存/利用率在 gpu_telemetry.csv。
物理 GPU 1/2/3 动态调度，每卡最多 12 个独立进程，无 DDP；保留 Exp18 的显存预留与容量恢复规则。
OOM 记录失败，停止新任务并等待在途任务结束；不改 batch、精度或参数。

恢复会复用完整且哈希校验通过的训练和测试结果。中断 trial 保留原样，不自动重跑。
确认所有 worker 已退出并检查日志后，可使用以下命令归档未完成的训练，再运行后台启动命令：

```bash
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp18b.recover --archive-incomplete
```

这是 trial 级恢复；重跑整个 trial，固定种子和配置。未完成的官方测试需人工检查，避免静默重复测试。

结果包括 trial_plan.csv、grid_per_seed.csv、grid_summary.csv、formal_per_seed.csv、summary.csv、
paired_differences.csv、best_hyperparameters.json、privacy_audit.json、gpu_telemetry.csv、final_report.md。
运行中不填造最终指标；完成后自动生成全部统计。逐 trial 配置、曲线、隐私、配对哈希、检查点在 trials/。

每次完整训练 ε=3、δ=1e-5，无采样放大；90 次训练的组合预算不能声明为 ε=3。
验证集视为公开；原始训练损失和裁剪诊断仅用于内部审计，不声明为额外 DP 发布。
