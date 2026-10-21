# Exp15b 最终报告

ε=8、δ=10⁻⁵ 仅为单次训练预算，不包含搜索、多种子训练及选参的组合隐私。历史官方评价集已使用过，本次不是新盲测。
选参使用内部验证；正式指标使用历史官方评价。十种子配对 Student-t 95% 区间未作多重比较校正，属于探索性统计证据。CV 正式训练包含原内部验证数据。
Momentum 与 Momentum+IID 的更新公式完全相同且均无偏差修正。IID 用身份策略和完整五轮参与敏感度 √5 重新校准；相关噪声沿用四频带策略。

|方案|任务|方法|lr|C/R|准确率|标准差|95% CI|交叉熵|
|---|---|---|---|---|---|---|---|---|
|A|cv|momentum|2|1|0.78868|0.00278|[0.78669, 0.79067]|0.78286|
|A|cv|momentum+iid|1|1|0.50115|0.00795|[0.49546, 0.50684]|1.84312|
|A|cv|momentum-bias|1|1|0.76596|0.00629|[0.76146, 0.77046]|0.79873|
|A|cv|stpex|1|1|0.75411|0.00719|[0.74896, 0.75926]|0.86499|
|A|cv|stpex2|1|1|0.75419|0.00718|[0.74905, 0.75933]|0.86396|
|A|cv|stpex-adam3|1|1|0.75405|0.00727|[0.74885, 0.75925]|0.86505|
|A|nlp|momentum|0.3|1|0.75023|0.00483|[0.74677, 0.75368]|1.17100|
|A|nlp|momentum+iid|0.3|1|0.74931|0.00469|[0.74596, 0.75267]|1.17251|
|A|nlp|momentum-bias|0.3|1|0.75000|0.00382|[0.74727, 0.75273]|1.17221|
|A|nlp|stpex|0.3|1|0.75573|0.00480|[0.75230, 0.75917]|2.09241|
|A|nlp|stpex2|0.3|1|0.75814|0.00392|[0.75534, 0.76094]|2.02616|
|A|nlp|stpex-adam3|0.3|1|0.75608|0.00371|[0.75343, 0.75873]|2.07414|
|B|cv|momentum|0.5|3|0.79142|0.00362|[0.78883, 0.79401]|0.72097|
|B|cv|momentum+iid|2|0.3|0.60736|0.01075|[0.59967, 0.61505]|1.37969|
|B|cv|momentum-bias|0.5|3|0.77549|0.00363|[0.77289, 0.77809]|0.77878|
|B|cv|stpex|0.5|3|0.76681|0.00418|[0.76382, 0.76980]|0.86181|
|B|cv|stpex2|0.5|3|0.76715|0.00425|[0.76411, 0.77019]|0.85773|
|B|cv|stpex-adam3|0.5|3|0.76675|0.00422|[0.76373, 0.76977]|0.86175|
|B|nlp|momentum|0.3|20|0.78670|0.01140|[0.77854, 0.79486]|0.53288|
|B|nlp|momentum+iid|0.3|5|0.75665|0.00999|[0.74950, 0.76380]|0.76673|
|B|nlp|momentum-bias|0.15|20|0.78360|0.00976|[0.77662, 0.79058]|0.53284|
|B|nlp|stpex|0.15|20|0.77947|0.00820|[0.77361, 0.78534]|1.34668|
|B|nlp|stpex2|0.2|20|0.78372|0.00934|[0.77704, 0.79039]|0.98858|
|B|nlp|stpex-adam3|0.15|20|0.78314|0.01043|[0.77568, 0.79061]|1.25361|

## 1. 相同超参数下的噪声机制效应

controlled_differences.csv 保存所有相同任务、lr、C、种子的 IID−相关噪声差。搜索/复核对照评价内部验证；只有两组正式选中相同配置时才有十种子官方受控对照，不能将单种子验证差当作十种子正式效应。
cv：10 个匹配配置，IID−相关噪声验证准确率差范围 [-0.65160, -0.11360]；这一区间是配置范围，不是置信区间。
nlp：15 个匹配配置，IID−相关噪声验证准确率差范围 [-0.27856, -0.00093]；这一区间是配置范围，不是置信区间。

## 2. 各自调参后的效果与 Momentum 偏差修正

A / cv：momentum+iid−momentum = -0.28753，95% CI [-0.29226, -0.28280]。
A / cv：momentum-bias−momentum = -0.02272，95% CI [-0.02585, -0.01959]。
A / nlp：momentum+iid−momentum = -0.00092，95% CI [-0.00456, 0.00273]。
A / nlp：momentum-bias−momentum = -0.00023，95% CI [-0.00264, 0.00218]。
B / cv：momentum+iid−momentum = -0.18406，95% CI [-0.18972, -0.17840]。
B / cv：momentum-bias−momentum = -0.01593，95% CI [-0.01740, -0.01446]。
B / nlp：momentum+iid−momentum = -0.03005，95% CI [-0.03941, -0.02068]。
B / nlp：momentum-bias−momentum = -0.00310，95% CI [-0.01278, 0.00659]。
Momentum 与 Momentum-Bias 的真实时间工作负载不同；主实验使用相同相关噪声滤波器，未针对未校正 Momentum 重新优化滤波器。

## 3. STP-EX2 是否稳定改善

cv/A：stpex2−stpex = 0.00008，95% CI [-0.00004, 0.00020]。
cv/A：stpex2−stpex-adam3 = 0.00014，95% CI [-0.00003, 0.00031]。
cv/B：stpex2−stpex = 0.00034，95% CI [0.00012, 0.00056]。
cv/B：stpex2−stpex-adam3 = 0.00040，95% CI [0.00017, 0.00063]。
cv 在两方案下相对于另外两种二阶矩均有正向配对区间支持：否。这不构成所有任务与配置上的普遍优势。
nlp/A：stpex2−stpex = 0.00241，95% CI [0.00042, 0.00440]。
nlp/A：stpex2−stpex-adam3 = 0.00206，95% CI [0.00034, 0.00379]。
nlp/B：stpex2−stpex = 0.00424，95% CI [-0.00398, 0.01247]。
nlp/B：stpex2−stpex-adam3 = 0.00057，95% CI [-0.00909, 0.01024]。
nlp 在两方案下相对于另外两种二阶矩均有正向配对区间支持：否。这不构成所有任务与配置上的普遍优势。

## 4. 排名变化与历史基线

cv/A：Exp15 momentum > momentum-bias > stpex2 > stpex > stpex-adam3；Exp15b momentum > momentum-bias > stpex2 > stpex > stpex-adam3 > momentum+iid；原五组排名是否变化：否。
nlp/A：Exp15 momentum-bias > momentum > stpex2 > stpex-adam3 > stpex；Exp15b stpex2 > stpex-adam3 > stpex > momentum > momentum-bias > momentum+iid；原五组排名是否变化：是。
cv/B：Exp15 momentum > momentum-bias > stpex2 > stpex > stpex-adam3；Exp15b momentum > momentum-bias > stpex2 > stpex > stpex-adam3 > momentum+iid；原五组排名是否变化：否。
nlp/B：Exp15 stpex2 > stpex-adam3 > momentum-bias > stpex > momentum；Exp15b momentum > stpex2 > momentum-bias > stpex-adam3 > stpex > momentum+iid；原五组排名是否变化：是。
cv 最佳 B/momentum 相对 Exp15/momentum 准确率均值差 0.02379（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp15/momentum-bias 准确率均值差 0.02546（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp15/stpex 准确率均值差 0.03731（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp15/stpex2 准确率均值差 0.03723（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp15/stpex-adam3 准确率均值差 0.03737（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp14d_A/dp-adam-iid-adam3 准确率均值差 0.19948（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp14d_A/dp-adam-sgd-bandinvmf-adam3 准确率均值差 0.06085（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp14d_A/dp-adam-momentum-bandinvmf-adam3 准确率均值差 0.02731（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp14d_A/dp-adam-momentum-bias-bandinvmf-adam3 准确率均值差 0.02060（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp14d_B/dp-adam-iid 准确率均值差 0.19714（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp14d_B/dp-adam-momentum-bandinvmf-adam3 准确率均值差 0.03482（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp14d_B/dp-adam-momentum-bias-bandinvmf-adam3 准确率均值差 0.02719（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp14d_B/dp-adam-sgd-bandinvmf 准确率均值差 0.09113（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp14d_B/dp-adam-sgd-bandinvmf-adam3 准确率均值差 0.07015（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp14d_B/dp-adam-sgd-bandinvmf-scale 准确率均值差 0.09858（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp9/dp-adam-momentum-bias-bandinvmf-scale 准确率均值差 0.01890（描述性跨实验差异）。
cv 最佳 B/momentum 相对 Exp9/dp-adam-momentum-bias-bandinvmf 准确率均值差 0.02418（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp15/momentum 准确率均值差 0.06686（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp15/momentum-bias 准确率均值差 0.06468（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp15/stpex 准确率均值差 0.08658（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp15/stpex2 准确率均值差 0.08108（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp15/stpex-adam3 准确率均值差 0.08521（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp15/momentum 准确率均值差 0.01124（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp15/momentum-bias 准确率均值差 0.00963（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp15/stpex 准确率均值差 0.01032（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp15/stpex2 准确率均值差 0.00700（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp15/stpex-adam3 准确率均值差 0.00768（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp14d_A/dp-adam-momentum-bandinvmf-adam3 准确率均值差 0.00654（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp14d_A/dp-adam-momentum-bias-bandinvmf-adam3 准确率均值差 0.00080（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp14d_A/dp-adam-iid-adam3 准确率均值差 0.02282（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp14d_A/dp-adam-sgd-bandinvmf-adam3 准确率均值差 0.00952（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp14d_B/dp-adam-iid 准确率均值差 0.02775（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp14d_B/dp-adam-momentum-bandinvmf-adam3 准确率均值差 0.00413（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp14d_B/dp-adam-momentum-bias-bandinvmf-adam3 准确率均值差 0.00195（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp14d_B/dp-adam-sgd-bandinvmf 准确率均值差 0.00814（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp14d_B/dp-adam-sgd-bandinvmf-adam3 准确率均值差 0.00665（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp14d_B/dp-adam-sgd-bandinvmf-scale 准确率均值差 0.00539（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp9/dp-adam-momentum-bias-bandinvmf 准确率均值差 0.00734（描述性跨实验差异）。
nlp 最佳 B/momentum 相对 Exp9/dp-adam-momentum-bias-bandinvmf-scale 准确率均值差 -0.00505（描述性跨实验差异）。

## 5. 搜索边界

cv/A/momentum+iid：lr=1、C/R=1；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。
nlp/A/momentum：lr=0.3、C/R=1；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。
nlp/A/momentum+iid：lr=0.3、C/R=1；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。
nlp/A/momentum-bias：lr=0.3、C/R=1；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。
nlp/A/stpex：lr=0.3、C/R=1；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。
nlp/A/stpex2：lr=0.3、C/R=1；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。
nlp/A/stpex-adam3：lr=0.3、C/R=1；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。
cv/B/momentum：lr=0.5、C/R=3；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。
cv/B/momentum+iid：lr=2、C/R=0.3；条件学习率边界=True，裁剪边界=True，需要未来独立实验验证；本次不追加网格。
nlp/B/momentum：lr=0.3、C/R=20；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。
nlp/B/momentum+iid：lr=0.3、C/R=5；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。
nlp/B/momentum-bias：lr=0.15、C/R=20；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。
nlp/B/stpex：lr=0.15、C/R=20；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。
nlp/B/stpex-adam3：lr=0.15、C/R=20；条件学习率边界=True，裁剪边界=False，需要未来独立实验验证；本次不追加网格。

## 6. 统计证据及局限

各自调参后的方法比较、两种方案比较均有十种子配对差及区间，见 paired_differences.csv。区间跨零表示当前十种子不足以明确方向；区间不跨零为未经多重比较校正的探索性支持。固定超参数搜索差及跨实验基线均按其实际样本范围作描述，不自动视为严格单因素或新盲测证据。
噪声标准差、边际协方差、完整协方差文件、裁剪比例、更新范数与实际 Momentum 工作负载效用见 noise_comparison.csv；非有限值及原因保留于各阶段结果。
本次合并 224 个候选，追加复核 92 次；正式配置去重后 240 次，其中复用 40 次。历史只读和结果哈希审计通过后才生成本报告。
