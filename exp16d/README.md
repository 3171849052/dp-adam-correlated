# exp16d

仅 CIFAR-100 / ViT-Tiny，AdaM³ 的 IID 和三个 B²LT workload。`adam3.py` 与 exp14a/optimizer.py 字节相同（exp14c/d 也使用此源码）；CV 数据、模型、梯度裁剪、噪声、训练循环及调度从 exp16c 复制，沿用 exp16a 协议。矩阵只读加载 exp16a 已验收的 T=225/250 文件，不求解。

训练：20 次搜索、16 次前二复核、40 次正式、公共配置额外 0–6 次；共 76–82 次完整训练。公共配置复用相同配置/种子的成功任务。另有 8 次两步 GPU smoke（四方法 × 两种 horizon），全部使用物理 batch 250 / 逻辑 batch 1000。CPU 真实模型 smoke 仅是小批诊断，不冒充生产批次验收。

先执行 CPU 测试并后台排队，等待 exp16c 的 pipeline_status 为 completed；随后自动 GPU smoke、搜索、复核、冻结、公共配置补齐、40 次正式、报告。若 exp16c 失败，本队列继续等待其恢复成功，不抢占 GPU。GPU 0–3 每卡最多一个本实验 worker；外部 GPU 进程占用时等待。worker 内部始终 cuda:0。不会改变批次、精度或参数重试。

从仓库根目录运行：

```bash
# 首次测试（已执行）
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp16d.stage1
# 启动 / 调度进程停止后的断点恢复（相同命令；flock 防重复启动）
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp16d.start_background
# 查看状态
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp16d.progress
tail -f exp16d/results/pipeline.log
```

成功 trial 由配置、源码、输入、summary、checkpoint 和日志 SHA256 校验后复用。失败保留目录并终止；部分任务需检查现场、保留归档后才可重跑，不自动删除或静默重试。尚在运行的后台进程不能重复启动。

全部产物位于 exp16d/results 与 exp16d/runtime。最终生成 search_results.json、selection_evidence.json、final_freeze.json、summary.csv、paired_differences.csv、controlled_differences.csv、final_report.md、final_audit.json。额外生成 Adam 历史对照、隐私校准、公共配置汇总。完整数值及 epoch 指标存放在各 trial summary.json。

各自最优超参数的比较不代表优化器本身因果效应。三种子公共配置与十种子正式结论分开。单次 ε=8 不代表整个搜索和重复实验联合 ε=8；历史测试集已使用，不声称新盲测。
