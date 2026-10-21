# Exp15c 最终报告

无噪声 Adam/Momentum 是非私有实验。Momentum-MF 和历史 Exp15b Momentum+MF 为单次训练 ε=8、δ=10⁻⁵；固定五轮参与、add/remove zero-out、不使用采样放大。搜索、复核及重复训练未声明组合隐私保证，内部验证假定公开；原始裁剪诊断是私有审计产物。

选参只用内部验证：搜索前二，复核两个种子，按三个种子的平均准确率和交叉熵排序。官方评价集历史上已被使用，这不是新盲测。A 固定 C=1；B 联合搜索。无噪声两方案共享同一学习率搜索。

|实验|方案|任务|方法|LR|C|准确率均值 ± 样本标准差 (%)|交叉熵|裁剪比例|Momentum workload MSE|
|---|---|---|---|---:|---:|---:|---:|---:|---:|
|Exp15c|A|cv|adam-no-noise|0.0003|—|83.977 ± 0.288|0.52966|—|—|
|Exp15c|A|cv|momentum-no-noise|0.05|—|84.819 ± 0.302|0.48787|—|—|
|Exp15c|A|cv|momentum-mf-native|2|1|79.609 ± 0.255|0.75138|0.8559|7.45311e-06|
|Exp15c|A|nlp|adam-no-noise|0.001|—|79.289 ± 0.908|0.74105|—|—|
|Exp15c|A|nlp|momentum-no-noise|0.1|—|78.727 ± 1.099|0.44792|—|—|
|Exp15c|A|nlp|momentum-mf-native|0.3|1|75.023 ± 0.411|1.17088|0.6430|7.94312e-06|
|Exp15c|B|cv|adam-no-noise|0.0003|—|83.977 ± 0.288|0.52966|—|—|
|Exp15c|B|cv|momentum-no-noise|0.05|—|84.819 ± 0.302|0.48787|—|—|
|Exp15c|B|cv|momentum-mf-native|2|1|79.609 ± 0.255|0.75138|0.8559|7.45311e-06|
|Exp15c|B|nlp|adam-no-noise|0.001|—|79.289 ± 0.908|0.74105|—|—|
|Exp15c|B|nlp|momentum-no-noise|0.1|—|78.727 ± 1.099|0.44792|—|—|
|Exp15c|B|nlp|momentum-mf-native|0.3|20|78.349 ± 1.194|0.54037|0.2336|0.00317725|
|Exp15b|A|cv|Exp15b-momentum-MF|2|1|78.868 ± 0.278|0.78286|0.8596|9.93296e-06|
|Exp15b|A|nlp|Exp15b-momentum-MF|0.3|1|75.023 ± 0.483|1.17100|0.6430|1.06591e-05|
|Exp15b|B|cv|Exp15b-momentum-MF|0.5|3|79.142 ± 0.362|0.72097|0.8587|8.93966e-05|
|Exp15b|B|nlp|Exp15b-momentum-MF|0.3|20|78.670 ± 1.140|0.53288|0.2417|0.00426364|

以下差值均为右组减左组，95% CI 为十个相同种子配对差的 Student-t 区间。准确率用百分点。CI 包含零时不作明确优越结论。

|方案|任务|左组|右组|准确率差 (百分点)|配对 95% CI|
|---|---|---|---|---:|---:|
|A|cv|adam-no-noise|momentum-no-noise|0.842|[0.618, 1.066]|
|A|cv|adam-no-noise|momentum-mf-native|-4.368|[-4.611, -4.125]|
|A|cv|adam-no-noise|Exp15b-momentum-MF|-5.109|[-5.339, -4.879]|
|A|cv|momentum-no-noise|momentum-mf-native|-5.210|[-5.346, -5.074]|
|A|cv|momentum-no-noise|Exp15b-momentum-MF|-5.951|[-6.096, -5.806]|
|A|cv|momentum-mf-native|Exp15b-momentum-MF|-0.741|[-0.853, -0.629]|
|A|nlp|adam-no-noise|momentum-no-noise|-0.562|[-1.224, 0.101]|
|A|nlp|adam-no-noise|momentum-mf-native|-4.266|[-4.919, -3.613]|
|A|nlp|adam-no-noise|Exp15b-momentum-MF|-4.266|[-4.927, -3.605]|
|A|nlp|momentum-no-noise|momentum-mf-native|-3.704|[-4.673, -2.736]|
|A|nlp|momentum-no-noise|Exp15b-momentum-MF|-3.704|[-4.669, -2.739]|
|A|nlp|momentum-mf-native|Exp15b-momentum-MF|0.000|[-0.109, 0.109]|
|B|cv|adam-no-noise|momentum-no-noise|0.842|[0.618, 1.066]|
|B|cv|adam-no-noise|momentum-mf-native|-4.368|[-4.611, -4.125]|
|B|cv|adam-no-noise|Exp15b-momentum-MF|-4.835|[-5.075, -4.595]|
|B|cv|momentum-no-noise|momentum-mf-native|-5.210|[-5.346, -5.074]|
|B|cv|momentum-no-noise|Exp15b-momentum-MF|-5.677|[-5.798, -5.556]|
|B|cv|momentum-mf-native|Exp15b-momentum-MF|-0.467|[-0.589, -0.345]|
|B|nlp|adam-no-noise|momentum-no-noise|-0.562|[-1.224, 0.101]|
|B|nlp|adam-no-noise|momentum-mf-native|-0.940|[-1.966, 0.085]|
|B|nlp|adam-no-noise|Exp15b-momentum-MF|-0.619|[-1.427, 0.188]|
|B|nlp|momentum-no-noise|momentum-mf-native|-0.378|[-1.673, 0.916]|
|B|nlp|momentum-no-noise|Exp15b-momentum-MF|-0.057|[-1.196, 1.082]|
|B|nlp|momentum-mf-native|Exp15b-momentum-MF|0.321|[-0.356, 0.998]|
|fixed lr=2.0, C=1.0|cv|Exp15b-momentum-MF|momentum-mf-native|0.741|[0.629, 0.853]|
|fixed lr=0.3, C=1.0|nlp|Exp15b-momentum-MF|momentum-mf-native|0.000|[-0.109, 0.109]|
|fixed lr=0.3, C=20.0|nlp|Exp15b-momentum-MF|momentum-mf-native|-0.321|[-0.998, 0.356]|

主要问题的结果（B 方案；A 方案及完整配对结果见上表）：

- cv 无噪声：Momentum 的十种子平均准确率较高；Momentum−Adam=0.842 个百分点，95% CI [0.618, 1.066]。配对区间不包含零。
- cv 独立调参的 Momentum-Matched MF：平均准确率超过历史 Momentum+MF，差值=0.467 个百分点，95% CI [0.345, 0.589]。配对区间不包含零。
- cv 无噪声 Adam−DP Momentum-MF 的差距=4.368 个百分点，配对 95% CI [4.125, 4.611]。
- cv 无噪声 Momentum−DP Momentum-MF 的差距=5.210 个百分点，配对 95% CI [5.074, 5.346]。
- cv 相同 LR/C 的单种子搜索对照：17 对中有 14 对提高准确率；差值范围 [-2.080, 5.920] 个百分点。这些配置并非独立重复，不据此计算十种子 CI。
- nlp 无噪声：Adam 的十种子平均准确率较高；Momentum−Adam=-0.562 个百分点，95% CI [-1.224, 0.101]。配对区间包含零，差异尚不明确。
- nlp 独立调参的 Momentum-Matched MF：平均准确率未超过历史 Momentum+MF，差值=-0.321 个百分点，95% CI [-0.998, 0.356]。配对区间包含零，不能确认稳定改善。
- nlp 无噪声 Adam−DP Momentum-MF 的差距=0.940 个百分点，配对 95% CI [-0.085, 1.966]。
- nlp 无噪声 Momentum−DP Momentum-MF 的差距=0.378 个百分点，配对 95% CI [-0.916, 1.673]。
- nlp 相同 LR/C 的单种子搜索对照：22 对中有 7 对提高准确率；差值范围 [-0.972, 0.710] 个百分点。这些配置并非独立重复，不据此计算十种子 CI。

历史 MF 仅从 Exp15b 读取，没有创建或重训历史方法。

任务验收：搜索/复核仅排除明确记录的非有限值，正式训练不排除任何种子。

|阶段|计划数|完成数|发散排除数|
|---|---:|---:|---:|
|search|63|62|1|
|review|30|30|0|
|formal|70|70|0|

排除原因（原始 summary、traceback 和训练日志全部保留）：

- search cv momentum-no-noise LR=1, C=None, seed=20261101：FloatingPointError: Nonfinite gradient；`exp15c/results/search/primary/cv/fc9900e31d31ed68386a/summary.json`。
- cv T=225：在相同 Momentum workload 上，重新校准后的理论 MSE 降低 24.73913%。
- cv T=250：在相同 Momentum workload 上，重新校准后的理论 MSE 降低 24.96587%。
- nlp T=310：在相同 Momentum workload 上，重新校准后的理论 MSE 降低 25.48036%。

MSE 为累计动量噪声的每坐标均方误差，包含 (1−β)²、C²、噪声校准和 1000² 归一化，未乘学习率平方；参数轨迹误差应再乘 LR²。mf_comparison.csv 固定 C=1。优化目标省略的 (1−β) 常数不影响滤波器最优解。

固定相同学习率和裁剪阈值的对照保存在 controlled_differences.csv（83 对）。搜索为单种子内部验证，复核只覆盖双方均已复核的配置，不能称作十种子官方机制效应。
如果双方选定的正式 LR/C 不同，固定超参数的十种子对照不存在；不额外训练历史方法。独立调参后的十种子比较仍列在配对表中。

完整交叉熵配对 CI 见 paired_differences.csv；最优超参数、十种子准确率/标准差/CI、交叉熵、裁剪比例和 MSE 见 summary.csv。
