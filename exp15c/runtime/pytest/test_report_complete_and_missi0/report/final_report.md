# Exp15c 最终报告

无噪声 Adam/Momentum 是非私有实验。Momentum-MF 和历史 Exp15b Momentum+MF 为单次训练 ε=8、δ=10⁻⁵；固定五轮参与、add/remove zero-out、不使用采样放大。搜索、复核及重复训练未声明组合隐私保证，内部验证假定公开；原始裁剪诊断是私有审计产物。

选参只用内部验证：搜索前二，复核两个种子，按三个种子的平均准确率和交叉熵排序。官方评价集历史上已被使用，这不是新盲测。A 固定 C=1；B 联合搜索。无噪声两方案共享同一学习率搜索。

|实验|方案|任务|方法|LR|C|准确率均值 ± 样本标准差 (%)|交叉熵|裁剪比例|Momentum workload MSE|
|---|---|---|---|---:|---:|---:|---:|---:|---:|
|Exp15c|A|cv|adam-no-noise|0.001|—|70.000 ± 0.000|0.80000|—|—|
|Exp15c|B|cv|adam-no-noise|0.001|—|70.000 ± 0.000|0.80000|—|—|
|Exp15c|A|cv|momentum-no-noise|0.001|—|70.000 ± 0.000|0.80000|—|—|
|Exp15c|B|cv|momentum-no-noise|0.001|—|70.000 ± 0.000|0.80000|—|—|
|Exp15c|A|cv|momentum-mf-native|0.001|1|70.000 ± 0.000|0.80000|0.1000|1e-05|
|Exp15c|B|cv|momentum-mf-native|0.001|1|70.000 ± 0.000|0.80000|0.1000|1e-05|
|Exp15c|A|nlp|adam-no-noise|0.001|—|70.000 ± 0.000|0.80000|—|—|
|Exp15c|B|nlp|adam-no-noise|0.001|—|70.000 ± 0.000|0.80000|—|—|
|Exp15c|A|nlp|momentum-no-noise|0.001|—|70.000 ± 0.000|0.80000|—|—|
|Exp15c|B|nlp|momentum-no-noise|0.001|—|70.000 ± 0.000|0.80000|—|—|
|Exp15c|A|nlp|momentum-mf-native|0.001|1|70.000 ± 0.000|0.80000|0.1000|1e-05|
|Exp15c|B|nlp|momentum-mf-native|0.001|1|70.000 ± 0.000|0.80000|0.1000|1e-05|
|Exp15b|A|cv|Exp15b-momentum-MF|0.001|1|70.000 ± 0.000|0.80000|0.1000|1e-05|
|Exp15b|B|cv|Exp15b-momentum-MF|0.001|1|70.000 ± 0.000|0.80000|0.1000|1e-05|
|Exp15b|A|nlp|Exp15b-momentum-MF|0.001|1|70.000 ± 0.000|0.80000|0.1000|1e-05|
|Exp15b|B|nlp|Exp15b-momentum-MF|0.001|1|70.000 ± 0.000|0.80000|0.1000|1e-05|

以下差值均为右组减左组，95% CI 为十个相同种子配对差的 Student-t 区间。准确率用百分点。CI 包含零时不作明确优越结论。

|方案|任务|左组|右组|准确率差 (百分点)|配对 95% CI|
|---|---|---|---|---:|---:|
|A|cv|adam-no-noise|momentum-no-noise|0.000|[0.000, 0.000]|
|A|cv|adam-no-noise|momentum-mf-native|0.000|[0.000, 0.000]|
|A|cv|adam-no-noise|Exp15b-momentum-MF|0.000|[0.000, 0.000]|
|A|cv|momentum-no-noise|momentum-mf-native|0.000|[0.000, 0.000]|
|A|cv|momentum-no-noise|Exp15b-momentum-MF|0.000|[0.000, 0.000]|
|A|cv|momentum-mf-native|Exp15b-momentum-MF|0.000|[0.000, 0.000]|
|A|nlp|adam-no-noise|momentum-no-noise|0.000|[0.000, 0.000]|
|A|nlp|adam-no-noise|momentum-mf-native|0.000|[0.000, 0.000]|
|A|nlp|adam-no-noise|Exp15b-momentum-MF|0.000|[0.000, 0.000]|
|A|nlp|momentum-no-noise|momentum-mf-native|0.000|[0.000, 0.000]|
|A|nlp|momentum-no-noise|Exp15b-momentum-MF|0.000|[0.000, 0.000]|
|A|nlp|momentum-mf-native|Exp15b-momentum-MF|0.000|[0.000, 0.000]|
|B|cv|adam-no-noise|momentum-no-noise|0.000|[0.000, 0.000]|
|B|cv|adam-no-noise|momentum-mf-native|0.000|[0.000, 0.000]|
|B|cv|adam-no-noise|Exp15b-momentum-MF|0.000|[0.000, 0.000]|
|B|cv|momentum-no-noise|momentum-mf-native|0.000|[0.000, 0.000]|
|B|cv|momentum-no-noise|Exp15b-momentum-MF|0.000|[0.000, 0.000]|
|B|cv|momentum-mf-native|Exp15b-momentum-MF|0.000|[0.000, 0.000]|
|B|nlp|adam-no-noise|momentum-no-noise|0.000|[0.000, 0.000]|
|B|nlp|adam-no-noise|momentum-mf-native|0.000|[0.000, 0.000]|
|B|nlp|adam-no-noise|Exp15b-momentum-MF|0.000|[0.000, 0.000]|
|B|nlp|momentum-no-noise|momentum-mf-native|0.000|[0.000, 0.000]|
|B|nlp|momentum-no-noise|Exp15b-momentum-MF|0.000|[0.000, 0.000]|
|B|nlp|momentum-mf-native|Exp15b-momentum-MF|0.000|[0.000, 0.000]|
|fixed lr=0.001, C=1.0|cv|Exp15b-momentum-MF|momentum-mf-native|0.000|[0.000, 0.000]|
|fixed lr=0.001, C=1.0|nlp|Exp15b-momentum-MF|momentum-mf-native|0.000|[0.000, 0.000]|

主要问题的结果（B 方案；A 方案及完整配对结果见上表）：

- cv 无噪声：两者相同 的十种子平均准确率较高；Momentum−Adam=0.000 个百分点，95% CI [0.000, 0.000]。配对区间包含零，差异尚不明确。
- cv 独立调参的 Momentum-Matched MF：平均准确率未超过历史 Momentum+MF，差值=0.000 个百分点，95% CI [0.000, 0.000]。配对区间包含零，不能确认稳定改善。
- cv 无噪声 Adam−DP Momentum-MF 的差距=0.000 个百分点，配对 95% CI [0.000, 0.000]。
- cv 无噪声 Momentum−DP Momentum-MF 的差距=0.000 个百分点，配对 95% CI [0.000, 0.000]。
- cv 相同 LR/C 的单种子搜索对照：1 对中有 0 对提高准确率；差值范围 [0.000, 0.000] 个百分点。这些配置并非独立重复，不据此计算十种子 CI。
- nlp 无噪声：两者相同 的十种子平均准确率较高；Momentum−Adam=0.000 个百分点，95% CI [0.000, 0.000]。配对区间包含零，差异尚不明确。
- nlp 独立调参的 Momentum-Matched MF：平均准确率未超过历史 Momentum+MF，差值=0.000 个百分点，95% CI [0.000, 0.000]。配对区间包含零，不能确认稳定改善。
- nlp 无噪声 Adam−DP Momentum-MF 的差距=0.000 个百分点，配对 95% CI [0.000, 0.000]。
- nlp 无噪声 Momentum−DP Momentum-MF 的差距=0.000 个百分点，配对 95% CI [0.000, 0.000]。
- nlp 相同 LR/C 的单种子搜索对照：1 对中有 0 对提高准确率；差值范围 [0.000, 0.000] 个百分点。这些配置并非独立重复，不据此计算十种子 CI。

历史 MF 仅从 Exp15b 读取，没有创建或重训历史方法。

- cv T=225：在相同 Momentum workload 上，重新校准后的理论 MSE 降低 24.73913%。
- cv T=250：在相同 Momentum workload 上，重新校准后的理论 MSE 降低 24.96587%。
- nlp T=310：在相同 Momentum workload 上，重新校准后的理论 MSE 降低 25.48036%。

MSE 为累计动量噪声的每坐标均方误差，包含 (1−β)²、C²、噪声校准和 1000² 归一化，未乘学习率平方；参数轨迹误差应再乘 LR²。mf_comparison.csv 固定 C=1。优化目标省略的 (1−β) 常数不影响滤波器最优解。

固定相同学习率和裁剪阈值的对照保存在 controlled_differences.csv（26 对）。搜索为单种子内部验证，复核只覆盖双方均已复核的配置，不能称作十种子官方机制效应。
如果双方选定的正式 LR/C 不同，固定超参数的十种子对照不存在；不额外训练历史方法。独立调参后的十种子比较仍列在配对表中。

完整交叉熵配对 CI 见 paired_differences.csv；最优超参数、十种子准确率/标准差/CI、交叉熵、裁剪比例和 MSE 见 summary.csv。
