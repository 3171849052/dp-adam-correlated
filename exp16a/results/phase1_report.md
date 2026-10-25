# Exp16a 阶段一验收实测

状态：passed。69 项测试通过，30 个矩阵通过逆关系/目标/敏感度/协方差/GDP 检查，46 次真实两逻辑步 smoke 完成。核心代码、平台代码、矩阵和证据 SHA256 见 phase1_acceptance.json。

所有矩阵最大逆关系绝对残差 6.939e-17；最大 GDP ε 核验误差 7.105e-15。完整时域噪声重放覆盖 T=225/250/310，另有经验协方差与 CUDA 缓冲重放。候选选择独立核验 27/27 通过，见 candidate_audit.json。

真实模型覆盖两任务各全部 14 方法，以及全部 14 个 CV 正式时域 smoke，另四次重复。初始化、分类头、训练顺序、增强/dropout 轨迹配对；重复模型逐张量完全一致。全部 FP32，CV 物理250、逻辑1000，NLP1000。官方评价使用数为零。

CV 实测最大 PyTorch allocated 峰值 8.069 GiB；NLP 3.814 GiB。双 NLP 与混合调度无 OOM。CPU/DTC 短程峰值不能代替完整时域内存测量。

求解状态必须区分数学正确性与收敛：9 个 BandInvMF 达到停止阈值；7 个 B²LT 与9个 DTC 达到预定1000步上限，另2个 B²LT 在可行回溯中停滞。后18个使用预注册规则保存预算内最低目标的可行候选，并未声称收敛或全局最优。DTC 的3个 Momentum-Bias 组合选择了 IID 基线。没有根据训练效用改动求解器或重试。

DTC 元数据 selected_parameters 保存最低目标的可行 L-BFGS 迭代参数；若 selected=iid，则最终矩阵明确为单位阵，以 NPZ 中 C/D 和 selected 标志为准。

|任务|方法|时域|重复|逻辑步|GPU|峰值 allocated GiB|实测秒|
|---|---|---|---:|---:|---:|---:|---:|
|cv|adam|search|0|2|0|7.427|19.72|
|nlp|adam|search|0|2|1|3.129|2.64|
|cv|adam-iid|search|0|2|2|7.759|19.49|
|nlp|adam-iid|search|0|2|1|3.393|2.07|
|cv|adam-mf-sgd-bandinvmf|search|0|2|3|7.780|20.37|
|nlp|adam-mf-sgd-bandinvmf|search|0|2|1|3.409|1.89|
|cv|adam-mf-sgd-b2lt|search|0|2|1|7.863|19.93|
|nlp|adam-mf-sgd-b2lt|search|0|2|1|3.475|2.26|
|cv|adam-mf-sgd-dtc|search|0|2|0|7.759|21.22|
|nlp|adam-mf-sgd-dtc|search|0|2|2|3.393|3.85|
|cv|adam-mf-momentum-bandinvmf|search|0|2|3|7.780|19.47|
|nlp|adam-mf-momentum-bandinvmf|search|0|2|2|3.409|2.75|
|cv|adam-mf-momentum-b2lt|search|0|2|1|7.863|19.72|
|nlp|adam-mf-momentum-b2lt|search|0|2|2|3.475|1.93|
|cv|adam-mf-momentum-dtc|search|0|2|3|7.759|20.37|
|nlp|adam-mf-momentum-dtc|search|0|2|2|3.393|3.23|
|cv|adam-mf-momentum-bias-bandinvmf|search|0|2|0|7.780|19.45|
|nlp|adam-mf-momentum-bias-bandinvmf|search|0|2|2|3.409|1.97|
|cv|adam-mf-momentum-bias-b2lt|search|0|2|1|7.863|19.24|
|nlp|adam-mf-momentum-bias-b2lt|search|0|2|2|3.475|2.07|
|cv|adam-mf-momentum-bias-dtc|search|0|2|3|7.759|20.47|
|nlp|adam-mf-momentum-bias-dtc|search|0|2|2|3.393|2.85|
|cv|adam-mf-sgd-stp-bandinvmf|search|0|2|0|7.986|19.10|
|nlp|adam-mf-sgd-stp-bandinvmf|search|0|2|2|3.750|2.10|
|cv|adam-mf-sgd-stp-b2lt|search|0|2|2|8.069|18.95|
|nlp|adam-mf-sgd-stp-b2lt|search|0|2|2|3.814|2.02|
|cv|adam-mf-sgd-stp-dtc|search|0|2|1|7.966|20.44|
|nlp|adam-mf-sgd-stp-dtc|search|0|2|2|3.733|3.00|
|cv|adam|final|0|2|0|7.427|17.42|
|cv|adam-iid|final|0|2|3|7.759|19.22|
|cv|adam-mf-sgd-bandinvmf|final|0|2|2|7.780|19.79|
|cv|adam-mf-sgd-b2lt|final|0|2|1|7.863|20.37|
|cv|adam-mf-sgd-dtc|final|0|2|0|7.759|19.82|
|cv|adam-mf-momentum-bandinvmf|final|0|2|3|7.780|18.95|
|cv|adam-mf-momentum-b2lt|final|0|2|2|7.863|20.51|
|cv|adam-mf-momentum-dtc|final|0|2|1|7.759|21.29|
|cv|adam-mf-momentum-bias-bandinvmf|final|0|2|0|7.780|20.17|
|cv|adam-mf-momentum-bias-b2lt|final|0|2|3|7.863|20.79|
|cv|adam-mf-momentum-bias-dtc|final|0|2|2|7.759|21.12|
|cv|adam-mf-sgd-stp-bandinvmf|final|0|2|1|7.986|19.03|
|cv|adam-mf-sgd-stp-b2lt|final|0|2|0|8.069|19.56|
|cv|adam-mf-sgd-stp-dtc|final|0|2|3|7.966|19.33|
|cv|adam|search|1|2|2|7.427|17.20|
|nlp|adam|search|1|2|2|3.129|1.68|
|cv|adam-mf-sgd-stp-b2lt|search|1|2|1|8.069|18.78|
|nlp|adam-mf-sgd-stp-b2lt|search|1|2|2|3.814|1.90|

Smoke 只评价64条内部样本，不是五轮训练或最终效用结果。完整流水线已经实际脱离终端启动；当前进度以 pipeline_status.json 和 *_status.json 为准。
