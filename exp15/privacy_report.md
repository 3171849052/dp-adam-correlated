# 相关噪声与隐私会计核验

每任务和训练长度读取 Exp9 已完成 Momentum-Bias trial 保存的四个 FIR 系数 d、策略矩阵 S 和设计工作负载 W。先核验历史 summary 的文件与矩阵内容 SHA256，再验证 S=D^{-1}、W 与 Exp9 Momentum-Bias 定义一致。每个 Exp15 trial 保存这些矩阵和该方法实际累计动量工作负载的独立哈希。设计 W 省略共同常数 (1-beta1)；真实工作负载包含该常数。

momentum-bias 和三个 STP-EX 变体的时间动量权重相同；momentum 不除以 1-beta1^t，工作负载不同。共用滤波器属于受控对照，不声称对 momentum 最优。STP-EX 的预条件器本身随带噪历史自适应；W 只描述 qtilde 到参数轨迹的一阶时间权重，不表示原始梯度的完整固定线性映射。

噪声在裁剪空间的梯度和上加入，再除以逻辑批大小 1000。标准高斯创新独立采样，保留的是创新历史，输出 z_t=sum_l d_l innovation_(t-l)。绝不把历史滤波输出当作创新。对实际阈值 C/R 分别校准创新标准差 sigma_sum = clip * sensitivity(S) / target_mu。相同阈值和矩阵可得到相同标准差，这是实际灵敏度相同的结果，未强制匹配数值；不同阈值按比例重新校准。

沿用 Exp9 的 add/remove zero-out 邻接、固定五轮参与和无抽样放大规则。令 G=S^T S、P_j={j,j+b,...,j+4b}，b 为每轮逻辑步数；sensitivity 使用 max_j sqrt(sum_{a,c in P_j} |G[a,c]|)。该 absolute Gram 计算对应 [JAX Privacy 的固定轮次参与灵敏度说明](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.matrix_factorization.sensitivity.fixed_epoch_sensitivity_for_X.html) 及其引用的式 (5)。零化一个固定槽位保留数据集长度和顺序，每次贡献范数上界为 C/R。GDP 的 target_mu 由 epsilon=8、delta=1e-5 反解；每步前缀使用完整参与日程的前缀策略重新计算，不把短程测试误记为完整预算。

预条件器只使用已加噪历史状态。每个逻辑批次冻结 d_(t-1)，累积期间不修改。stpex2 二阶矩只接收 qtilde，stpex 接收 d_(t-1)*qtilde，stpex-adam3 接收已加噪输入生成的未校正 u；不存在由未加噪 q 直接更新后续分母的路径。

数值验证包括：全部三种有效矩阵长度 225/250/310 的逆关系、不同裁剪阈值的完整 epsilon=8 反算、前缀预算、动量真实轨迹、30,000 条独立 FIR 路径的协方差、随机流精确复现，以及小模型逐样本显式梯度与两次反向裁剪的一致性。实际测试结果见 `results/tests.log`；真实模型结果见 `results/phase1_acceptance.json` 和各 trial 的 calibration。

这核验的是继承机制的实现一致性，未提供新的自适应矩阵机制定理。单次模型训练预算不自动包含超参数搜索、公开验证集以外的数据发布、十次运行的组合开销。未加噪损失/裁剪统计和梯度范数属于本地私有审计数据；保存的创新种子及 RNG 状态也不能随 DP 模型作为公开随机性发布。理论随机机制假设高斯创新独立且保密；固定种子实验用于可复现的机制数值研究，不把公开种子下的确定性执行宣称为 DP 发布。
