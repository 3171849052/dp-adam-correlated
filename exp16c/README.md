# exp16c：CV B²LT / IID 扩展

所有运行均使用已有 `curve` 环境；新产物仅写入本目录。历史模块复制为本地模块，避免导入历史包 initializer 而写入其 runtime。没有矩阵求解器；矩阵直接读取 exp16a 已验收 T=225/250 文件并验证 SHA256。

## 启动、进度、恢复

在仓库根目录执行：

```bash
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp16c.progress
cat exp16c/results/background.json
tail -f exp16c/results/pipeline.log
```

首次验收与后台启动（已经由本次实施执行，无需重复）：

```bash
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp16c.stage1 > exp16c/results/stage1.log 2>&1
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp16c.start_background
```

恢复入口为同一 `start_background` 命令。独立 session + flock 单实例；关闭终端后继续运行。物理 GPU 0/1/2/3 各至多一个 CV worker，worker 仅看到 `cuda:0`。外部占用的 GPU 会等待，不抢占其他实验进程。不使用 DDP。

完成 trial 校验配置、代码、输入、矩阵及产物 SHA256 后复用。数值发散的搜索/复核记录并排除；工程错误/OOM/正式失败停止新派发，已启动任务收尾。不静默改参数或自动重试。中断的未完成 trial 保留现场；排查后将该 trial 目录移入 `exp16c/results/recovery/` 再运行恢复入口，可按原配置完整重跑；不得移动尚在运行的 worker 目录。不是步级 checkpoint 恢复。

## 固定协议和计划

仅 CIFAR-100 + 本地 ViT-Tiny，禁止下载。exp9 固定内部划分与 exp16a 的初始化、增强、固定 epoch 顺序；FP32，全参数；逻辑 batch=1000，物理=250。搜索/复核 45000/5000、五轮225步，正式 50000/10000、五轮250步。搜索和 smoke 不打开官方 test。搜索 20261101，复核 20261102/03，正式 20261111–20。每次训练 ε=8、δ=1e-5、固定五轮参与、GDP、无采样放大。

- 新搜索 25：Adam 4、STP-EX 6、Momentum 15；`search.additions()` 明确列出用户指定组合。
- 复用 exp16b 27 次 B²LT CV 搜索，以及 exp16a 的相关搜索/复核/正式结果。只取本任务范围内方法。
- 六个待调参方法分别合并候选取前二，追加两个复核种子；已有有效 trial 去重。两个 Adam 另保留历史三种子最优作基准，严格改善准确率/次级损失才替换。
- 新复核 16–24 次；新正式 40–60 次。STP-EX 和三个 Momentum 各必须真正训练10种子；Adam 每发生一次最优替换增加10次。其他 Adam B²LT 与 Adam-IID 仅复用历史基线。最终表包含9方法×10种子=90条正式结果。
- Momentum 原样复用 exp15c：`m=.9*m+(1-.9)*g_noisy; p+=-lr*m`；无偏差修正、Nesterov、weight decay。三个 Momentum 的唯一区别为噪声策略/矩阵，搜索网格相同。
- exp16b 原版 STP-EX 保留其第一矩偏差修正、前一轮 vhat 分母、scaled clipping/noise 和原版二阶矩更新，不与 exp16a STP 混用。

单次 ε=8 不代表搜索和所有重复运行的组合预算。内部验证按历史协议假设公开，裁剪诊断为内部审计材料。

## 输出

`phase1_acceptance.json`、`unit_tests.log`、`smoke_results.json` 为验收；`*_status.json`、`pipeline_status.json`、`scheduler.jsonl`、`gpu_telemetry.jsonl` 提供运行现场。`progress` 命令刷新 `progress.json`。每个 trial 保留 PID、日志、配置、逐步诊断、逐轮指标、checkpoint、完成清单。

流水线自动生成 `top2.json`、`selection_evidence.json`、`final_freeze.json`，随后实际执行正式训练。完成后生成 `summary.csv`、`formal_per_seed.csv`、`paired_differences.csv`、`paired_per_seed.csv`、`momentum_comparison.csv`、`controlled_momentum.csv`、`matrix_errors.csv`、`momentum_workload_errors.csv`、`final_report.md`、`final_audit.json`。正式统计采用样本SD与 Student-t 95% CI；配对前核验初始化、数据顺序和非噪声随机流。调参后比较与共同超参数的内部验证受控比较分开解释。
