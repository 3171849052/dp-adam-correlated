# Exp15d：STP-EX 稳定常数搜索

仅搜索 stpex、stpex2、stpex-adam3 的 eps_scale，完整训练实现从 Exp15b 复制并以源码等价测试锁定。Momentum-Bias 四频带 MF 不变，不采用 Exp15c Momentum workload。全部新增文件、临时目录与输出在 exp15d/；数据、权重、历史结果只读，完全离线。

使用 `/home/longt29/miniconda3/envs/curve/bin/python -B -m exp15d.audit` 准备只读审计；再运行 `exp15d.checks` 和 `exp15d.smoke`。二者通过后运行 `exp15d.start_background`，独立会话继续执行搜索→复核→正式十种子→统计报告。所有模块通过 `python -B -m` 从仓库根目录执行。

物理 GPU 0/1/2/3 动态调度，每卡一个训练进程；每个 worker 仅见自身物理卡，内部使用 cuda:0。查看 `results/pipeline.log`、`results/pipeline_status.json`、`results/background.json` 与各阶段 `*_status.json`。完成记录需配置、训练代码及全部产物哈希一致才能复用；未完成或工程失败任务停止并要求排查，不自动重试。可用同一启动命令恢复已成功的任务。

历史 eps 配置全部重新训练。历史 Exp15b 正式结果只作报告基线，不训练任何历史方法。已有官方评价集使用记录，本次不是新盲测。

最终自动生成 search_results.json、review_results.json、formal_results.json、summary.csv、paired_differences.csv、controlled_differences.csv、eps_sensitivity.csv、curves/*.png、final_report.md、final_audit.json。正式结果只在实际完成后生成。

详见 [protocol.md](protocol.md)。
