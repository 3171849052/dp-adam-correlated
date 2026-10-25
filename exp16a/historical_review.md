# 只读来源核对

已阅读 Exp9 config/data/model/clipping/noise/privacy/train/search/launcher、实验 README、platform_validation 和既有 smoke/正式结果；Exp14a/b/c/d README、协议及 Exp14c 正式报告；Exp15/b/c/d README、协议、Exp15c 原生 Momentum 求解/训练和最终报告、Exp15d 最终报告。源哈希见 source_provenance.json；随附 stp-ex.pdf 转录在 stp-ex.txt，第 4 页图核对了 B²LT 起点 ω=−(1−θ)/(2Q)。

Exp9 的 Scale 用历史 vhat 缩放逐样本梯度，scaled clip/noise 后撤销缩放，交给普通 Adam。Exp15 STP-EX 的一阶动量位于预条件化空间，最终直接按该动量更新参数，二者不同；本实验仅复用 Exp9。

Exp9 以及后续部分实验只用了 BISR 或 Nelder-Mead；不能把它们标作本次要求的论文 L-BFGS 求解。旧 Momentum workload=(1−β^(lag+1))/(1−β)，本次严格为 1−β^(lag+1)。旧 Momentum-Bias 使用完整非 Toeplitz 定义，本次同样保留；不额外补乘或补除任何整体常数。

Exp14 更改 Adam 为 AdaM³（二阶矩使用未修正动量平方）；既有正式结果显示优化器和各自选参会影响比较。本次全部用标准 Adam，历史结果不作为新结果或新选参证据。Exp15c 为 Momentum 优化 workload，不能直接当作本次 Adam 最优矩阵。

继承 Exp9 CV 45000/5000、正式50000/10000；NLP 62000/5349、官方872；相同 split seed20261008、训练种子、物理/逻辑批次和 FP32。直接只读使用历史 split/token 文件并核对其资产 SHA256；无历史训练结果复用。正式 CV 的内部验证合入训练集，因此正式报告使用官方 test。

GDP 使用完整固定五轮参与敏感度。PDF §5 的保守 εδ 校准式不与 Exp9 GDP 噪声幅度混用；仅移植 PDF 的矩阵/障碍/候选选择/在线生成部分。PDF 中 workload 对 STP-EX 的精确线性展开不适用于本实验自适应 Adam，报告明确将矩阵误差作为设计指标。

开发期间发现 SciPy 的不可行值回退会产生首步假收敛，初始产物保存在 results/development_solver_attempt1，仅供排查。正式求解使用明确可行性检查、Armijo 回溯和 L-BFGS 双循环，候选只取实际接受的可行迭代及规定基线。达到固定迭代上限或回溯停滞时保留真实 solver 状态和候选，不宣称全局最优或收敛。

还修正了 torchvision 的非评估数据访问细节：CIFAR100 默认完整性检查会同时散列 train/test。Exp16a 保留相同数据读取和变换，只将完整性检查限制为当前实际请求的 split，确保搜索和 smoke 不打开官方 test 文件；独立测试覆盖此约束。
