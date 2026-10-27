# 在 exp17 后使用 GPU 2、3 恢复 exp16b

2026-10-27 按用户要求将 exp16b 剩余 71 个粗搜索试验排在 exp17 后面，使用物理 GPU 2、3，后台独立 session 运行。保留并完整校验此前完成的 37 个结果。昨天中断的试验 `359aaae1240908c02cb6` 已归档，恢复时从第 0 步重跑。

等待条件是 exp17 **整个 pipeline 成功完成，包括 final 和 ablation**，且 exp17 调度器释放锁并退出。训练之间的短暂空闲不会触发 exp16b。exp17 若失败或尚未完成，队列继续等待；等待原因写入状态文件。

实际运行后每张卡最多一个 CV 进程或两个 NLP 进程，同卡不混跑 CV 和 NLP。所有 worker 各自只看见一张物理 GPU；GPU 0、1 不参与此队列。外部进程占用 GPU 2 或 3 时，对应卡等待。工程故障保留现场并停止新任务派发，不自动重试。

查看后台 PID、等待状态、进度和日志：

```bash
cat exp16b/results/background.json
cat exp16b/results/pipeline_status.json
cat exp16b/results/queue_gpus_2_3.json
cat exp16b/results/progress.json
tail -f exp16b/results/pipeline.log
```

需要重新启动同一队列时，在确认原后台进程已退出、没有未处理的中断 trial 后执行：

```bash
/home/longt29/miniconda3/envs/curve/bin/python -B exp16b/runtime/resume_gpus_2_3.py --check
/home/longt29/miniconda3/envs/curve/bin/python -B exp16b/runtime/resume_gpus_2_3.py --start
```

原 `pipeline.lock` 单实例锁覆盖整个等待及训练过程。队列运行时，旧 GPU 0 恢复入口也无法启动第二个调度器。

此次仅在 `exp16b/runtime/` 添加部署适配器，保留验收过的训练源文件和哈希验证。调度适配只将设备列表改为 GPU 2、3，并指定适配 worker；训练适配只修改单 GPU 准入条件。每个新 trial 的 `gpu_migration.json` 记录实际执行的函数源码、哈希和适配器哈希，并纳入原 completion 完整性清单；实际物理 GPU 也会写入 config、process 和 summary。队列级部署记录位于 `results/gpu_transition.json`。

本次已通过原验收及 37 个完整结果的恢复校验，以及 7 项不启动 CUDA 的队列检查；检查日志为 `runtime/queue_gpus_2_3_checks.log`。后台启动前的状态另存于 `runtime/before_queue_2_3_*/`，此前的中断现场归档继续保留。
