# 实施前历史核对

已阅读 exp16a 的 config/data/cv_model/clipping/optimizer/noise/privacy/matrices/train/search/scheduler/pipeline、protocol.json 和已完成报告；exp16b 的原版 STP-EX optimizer、只读矩阵实现、粗搜索协议及全部27条 B²LT CV结果；exp15c 的 Momentum optimizer、protocol.md 和最终报告。

- exp16a 两个待扩展 Adam 方法的历史最优均为 lr=.003、clip=17.3205080757；正式准确率分别为 76.995±0.538%（SGD workload）和 76.921±0.566%（Momentum-Bias workload）。这些是历史结果，不是 exp16c 新结果。
- exp16b B²LT CV 27 次均完成；单种子第一名为 (.15,10,3)，内部验证 .7794，第二名 (.15,10,1)，内部验证 .7790。只读取候选，不能直接据此宣称三种子最优。
- exp15c 的 Momentum-MF CV 联合调参最优为 (2,1)，历史正式均值79.609%；其矩阵是另一种已优化 FIR，不能用来替代本实验 B²LT 正式训练。这里只复用未修正 Momentum 的精确更新顺序。
- exp16a 标准 clipping/Adam logical batch、噪声与GDP保持本地复制实现；exp16b 的 STP-EX logical batch 保留 scaled q 更新、无最后一次除分母；exp15c Momentum logical batch 保留原梯度空间裁剪、加噪后除以1000。
- 不导入历史包，避免其 `__init__` 修改历史 runtime。历史 trials 通过原 completion 清单、配置和代码 SHA256、已完成步数、矩阵来源核验后，只读引用其 source 路径。
- 历史相关有效搜索148条（exp16a121、exp16b27），复核20条。正式基线仅在全配置和种子匹配时复用；在最优冻结后才读取正式 trial。
