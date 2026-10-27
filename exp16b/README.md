# exp16b：原版 STP-EX 的 Momentum-Bias 粗搜索

2026-10-27 运行安排更新：剩余试验后台排在 exp17 全流程（含消融）之后，使用物理 GPU 2、3。当前启动入口及状态查看方法见 [GPU 2、3 队列说明](RESTART_GPUS_2_3.md)；下文 GPU 0 启动命令保留为原始方案记录。

仅两个方法：`stpex-mf-momentum-bias-bandinvmf`、`stpex-mf-momentum-bias-b2lt`。
原版公式来自 `exp15/optimizer.py`、`exp15d/optimizer.py`、`exp15d/protocol.md` 和仓库 `stp-ex.pdf`（另参 `exp16a/stp-ex.txt`）：

- 冻结上一轮二阶矩：`d = sqrt(v/(1-β₂^(t-1))) + eps_scale`，首步 `d=eps_scale`。
- 每样本在 `g/d` 空间按整个参数向量的 L2 范数裁剪。`q=(Σ clip(g/d,R)+z_sum)/1000`。
- `u=0.9u+0.1q`；`x=x-lr*u/(1-0.9^t)`；**参数更新不再除分母**。
- `v=0.999v+0.001(d*q)^2`。本轮新二阶矩仅供下一步使用。

四套矩阵直接读取 exp16a 已验收 Momentum-Bias、T=225/310 的 BandInvMF/B²LT 文件，同时核验验收清单、元数据及文件 SHA256；无矩阵求解器。保留 exp16a 工作矩阵原尺度（线性动量的全局 1-β₁ 因子不另改矩阵）。五次固定参与、add/remove zero-out、无抽样放大；GDP 解出 μ，`sigma_sum=R*Sens(C)/μ`，单次训练 ε=8、δ=1e-5。ε 不是 eps_scale；108 次搜索整体不具有单次 ε=8 保证。原始梯度诊断限内部审计，内部验证按历史协议处理。

继承 exp16a 的离线模型、FP32、初始化、增强、固定顺序和随机流；数据源 data/，预训练参数 cache/，沿用 exp9 经资产清单校验的内部划分及 SST-2 预分词缓存。没有官方测试集读取路径，无下载。CV：45000 条、5轮225步，逻辑1000/物理250；NLP：62000条、5轮310步，逻辑/物理1000。

冻结网格 `results/coarse_grid.json`：CV lr=[.15,.5,1.5]，clip=[1,3,10]，eps_scale=[.1,1,3]；NLP lr=[.05,.15,.3]，clip=[5,20,40]，eps_scale=[.0001,.001,.01]。每方法每任务27个，种子20261101，共108次完整五轮。没有细网格、额外种子或正式训练入口。

## 启动、恢复与进度

仓库根目录、已有 curve 环境：

```bash
CUDA_VISIBLE_DEVICES=0 /home/longt29/miniconda3/envs/curve/bin/python -B -m exp16b.stage1 > exp16b/results/acceptance.log 2>&1
CUDA_VISIBLE_DEVICES=0 /home/longt29/miniconda3/envs/curve/bin/python -B -m exp16b.start_background
cat exp16b/results/background.json
cat exp16b/results/progress.json
cat exp16b/results/pipeline_status.json
tail -f exp16b/results/pipeline.log
```

第二条在验收通过后实际执行，独立 session、文件日志、继承 flock 单实例锁。恢复同样执行第二条：仅复用配置、代码、输入、矩阵、checkpoint、日志及完成清单哈希全部一致的已完成结果。明确 non_finite 记录并排除；OOM、其他工程故障和未完成现场停止并保留，不自动覆写或重试。支持已完成 trial 粒度的断点恢复，不承诺中断训练的步级续训。恢复前先检查 background.json PID 及已有 worker；锁冲突拒绝第二调度器，外部占用 GPU 0 时等待。

调度仅 CUDA_VISIBLE_DEVICES=0，CV 至多1进程，NLP至多2独立进程，二者不混跑，无DDP。验收包含 CPU/GPU 公式、逐样本裁剪与累积、重复 token、完整时域在线噪声、经验协方差、隐私校准、四套矩阵检查；随后每任务每方法两逻辑步真实模型 smoke，另各重复一次检查逐张量完全复现（共8次，不计入108次）。

每次训练目录 `results/search/{cv,nlp}/<ID>/` 保存 config、逐步 diagnostics、逐轮 metrics、summary、checkpoint、训练顺序、矩阵、process/PID、日志及完成哈希清单。失败保存 traceback 和已产生指标。`coarse_status.json`、`progress.json`、`scheduler.jsonl`、`gpu_telemetry.jsonl` 提供运行状态。四张 `ranking_*.csv` 包含全部27候选，完成者按末轮内部准确率降序、验证损失升序、ID升序排名，其他状态不参与排名。`best_candidates.json` 和 `coarse_report.md` 随进度更新边界诊断，当前候选仅为单种子暂定结果，不冻结正式最优配置。

所有新文件及缓存仅在 exp16b/；历史模块复制后改为本地导入，避免运行历史包的 initializer。
