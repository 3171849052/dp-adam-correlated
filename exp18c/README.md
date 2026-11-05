# Exp18c 学习率扩展

在 Conda `curve` 中运行；所有新增文件与缓存均位于本目录。

- `common.py`、`setup.py`：固定九个新候选，复制并校验 Exp18b 的五份初始化和三份矩阵，复算相同 GDP 隐私参数，自动检测四张 GPU 的编号和 UUID。
- `train.py`：沿用 Exp18b 训练流程，直接复用 Exp18 CNN、Update、OnlineNoise；仅访问训练/验证数据。保存 162 步训练曲线、3 次验证、配置、隐私参数、配对哈希及检查点。
- `scheduler.py`：复用 Exp18 的 admission_limit，按显存容量轮转分发独立进程；UUID 绑定 GPU，每进程只可见一张卡。保留 1536 MiB，每进程预算由六次冒烟实测校准，上限每卡 12 进程。
- `accept.py`、`tests/`：协议与算法测试、六次最高学习率完整冒烟、四卡分配、初始化/排列/创新流配对检查。通过后冻结代码、配置、矩阵及初始化哈希。
- `pipeline.py`、`start_background.py`：持有文件锁，创建独立会话，stdin=/dev/null，自动执行 90 次正式训练、验证集选参、冻结选择、30 个选定检查点的官方测试和统计报告。
- `report.py`：复用 Exp18b 五种子统计，生成两轮的 IID/B2LT/平均验证曲线比较、共同 LR、搜索边界与配对差值；合并网格分析不改变 Exp18c 的正式选参范围。

```bash
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp18c.accept > exp18c/acceptance.log 2>&1
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp18c.start_background
```

验收后不得修改冻结代码或协议。`background.json` 记录 PID、命令与日志；`status.json` 记录阶段和活动任务；`gpu_telemetry.csv` 与 `gpu_processes.jsonl` 记录四卡资源；每个任务日志在 `trials/<id>/worker.log`。运行完成后 `final_report.md` 自动更新，全部要求的结果文件自动生成。重复启动时文件锁阻止双重运行；已完整且哈希验证通过的任务可复用，未完成任务保留现场并停止，不自动更改参数或重复测试。

Exp18b/Exp18/data 的文件哈希保存在 `upstream_manifest.json`，结束时核对全部文件并生成 `readonly_verification.json`。不下载数据、不优化矩阵、不更改原始算法。每次完整训练 ε=3、δ=1e-5、GDP 无采样放大；该预算不表示多次训练组合预算也是 ε=3。原始训练曲线仅为内部审计数据。
