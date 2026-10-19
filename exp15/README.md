# Exp15

五种优化器、CV/NLP 两类任务、两种搜索方案。所有新增文件及运行产物均位于 `exp15/`，历史实验和原始数据只读。

从仓库根目录运行，使用已有的 `curve` 环境：

```bash
/home/longt29/miniconda3/envs/curve/bin/python -B -m pytest exp15/tests -q -p no:cacheprovider
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp15.smoke
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp15.acceptance_report
/home/longt29/miniconda3/envs/curve/bin/python -B -m exp15.pipeline
```

`smoke` 对 CV 搜索/正式两种步数、NLP 共执行 15 次两逻辑步测试，另复跑 CV/NLP 的 stpex2，核验逐张量完全复现及五方法随机流配对。只评价 64 个内部验证样本。验收通过后写 `results/phase1_acceptance.json`。烟雾测试 lr=0.01、C/R=1 只为公式和管线验收，不用于选参。

阶段二自动顺序：50 次初始摸底（每次 20 逻辑步）→ 最多两轮学习率边界探测 → 记录提案并冻结 → 90 个完整五轮搜索候选 → 按两方案分别选参 → 最多 200 次正式配对训练（完全相同配置去重）→ 统计报告。正式候选每任务、每方法为三学习率 × 三阈值，方案 A 取阈值 1 的三个点，复用方案 B 的结果。冻结后不追加、删除或修改正式候选。

初始摸底以 Exp9 Momentum-Bias Adam 的真实学习率为量级参照，尝试 1×/10×/100× 及 0.1/1/10 裁剪阈值。算法更新不同，历史最优配置不作为 Exp15 最优配置。最终三学习率为摸底中 C/R=1 最优稳定点的 1/3、1、3 倍；若粗网格边界胜出，先进行预先规定的最多两轮十倍边界探测。早期四步 CV 探针仍接近随机分类，已在冻结前统一延长为 20 步，初步记录见 `results/preliminary_4step/` 和 `results/protocol_revision.md`。短程摸底不证明长程最优。最终范围及依据以实际 `grid_proposal.json` 为准。

`results/pipeline_status.json` 记录当前阶段，`*_status.json` 和 `scheduler.jsonl` 记录命令、GPU、PID 和错误。每个 GPU 最多一个训练进程，外部占卡则等待。完整训练成功结果可校验后复用，不覆盖已有结果，不自动重试。摸底/搜索的非有限值可作为失败候选记录；算法、校准、文件哈希、OOM 和其他异常停止流水线。

单次训练入口示例（已经执行的结果不会覆盖）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/longt29/miniconda3/envs/curve/bin/python -B -m exp15.train --trial-json '{"task":"cv","method":"stpex2","lr":0.01,"clip":1.0,"stage":"smoke","seed":20261099}'
```

种子从 `exp9/config.py` 的 AST 读取，不导入历史 initializer。正式种子实际为 20261111–20261120，搜索为 20261101；摸底/验收使用不同种子。最终评价沿用历史官方评价集；只有内部验证集参与搜索选择。最终 CV 用全部 50,000 个训练样本，因此内部验证不能作为最终独立验证；NLP 继续使用历史 62,000 条训练子集。

优化器和历史差异见 [differences.md](differences.md)，隐私、噪声与随机流说明见 [privacy_report.md](privacy_report.md)，阶段一实测见 `results/phase1_report.md`。完整统计只在正式运行全部成功后生成 `results/final_report.md`。
