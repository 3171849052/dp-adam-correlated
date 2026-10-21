# Exp15c

三个方法：`adam-no-noise`、`momentum-no-noise`、`momentum-mf-native`。前两组为无裁剪、无噪声的非私有全参数 FP32 微调。第三组使用 Exp15b 的无偏差修正 Momentum、原始梯度空间裁剪，以及重新优化并校准的 Momentum workload 四频带 MF。没有新增 Momentum-Bias 训练组。

为避免导入 Exp15b 时将缓存路径重设到其目录，数据、模型、隐私和 FIR 噪声实现从 Exp15b 复用为本目录的隔离副本；Ghost clipping 只保留原始梯度空间实现。Momentum 更新逐张量与 Exp15b 验证；Adam 直接使用 `torch.optim.Adam(eps=1e-8)`。所有新增产物在本目录，历史文件只读，数据和模型禁止下载。

使用现有 `curve` 环境，从仓库根目录运行：

```bash
conda run -n curve python -B -m exp15c.bandinvmf
conda run -n curve python -B -m exp15c.checks
conda run -n curve python -B -m exp15c.smoke
conda run -n curve python -B -m exp15c.start_background
```

最后一条通过脱离终端的独立会话启动完整流水线，自动执行搜索 → 复核 → 正式十种子 → 报告。仅使用物理 GPU **1、2、3**，每卡同时最多一个训练任务。worker 通过 `CUDA_VISIBLE_DEVICES=<物理编号>` 隔离，内部统一使用 `cuda:0`；GPU 0 不用于 Exp15c。外部占卡时等待。

搜索 63 次（非私有 24，DP 39）；复核最多 32 次；正式最多 80 次。A/B 相同配置跨方案去重。已完成并通过代码/产物哈希检查的训练不重跑。中断后执行同一后台入口恢复已完成任务；搜索/复核中明确记录为 `non_finite` 的候选保留现场、排除排名并继续，其已有记录在恢复时直接复用。OOM、普通程序错误、哈希或校准异常仍停止且不自动重试；正式训练的任何失败（包括非有限值）均严格停止。没有有效候选时明确停止。

查看进度：

```bash
cat exp15c/results/background.json
cat exp15c/results/pipeline_status.json
conda run -n curve python -c "import json,collections; d=json.load(open('exp15c/results/search_status.json')); print(collections.Counter(v['status'] for v in d.values()))"
tail -f exp15c/results/pipeline.log
```

每个阶段的 `*_status.json` 保存全部任务状态、PID、物理 GPU 和任务日志路径；逐步诊断在各 trial 的 `diagnostics.jsonl`。调度事件在 `scheduler.jsonl`。

`results/mf/` 在训练前保存 225/250/310 步的滤波器、策略矩阵、workload、优化元数据及独立校准；`mf_comparison.csv` 立即可用。阶段完成后写入 `search_results.json`、`review_results.json`、`formal_results.json`；全部完成且审计通过后生成 `summary.csv`、`paired_differences.csv`、`controlled_differences.csv` 和 `final_report.md`。尚未完成的阶段不制造结果或最终准确率。

历史比较只读取 Exp15b 的 Momentum+MF 搜索、复核、正式结果和选参文件。固定 LR/C 比较与独立调参比较分别报告；前者没有匹配正式配置时仅有内部验证证据，不补训历史方法。官方评价集历史上已经使用，不称为新盲测。单次 DP 训练 ε=8、δ=10⁻⁵，不声明整个搜索或重复实验的组合隐私保证。详见 [protocol.md](protocol.md)。

本次恢复保留原始 smoke 验收，按训练核心哈希校验其有效性；调度/搜索/报告逻辑改动需当前回归测试通过。冻结网格和成功 trial 的配置、核心代码、结果产物哈希继续严格检查。`results/recovery/` 保存恢复前的状态、测试证据及产物清单。最终报告和 `final_audit.json` 记录各阶段计划数、完成数、发散排除数和逐项原因；排除项不参与统计。
