# Exp15d 最终报告

历史官方评价集已使用过，本次不是新盲测。选参仅使用内部验证；未根据官方结果修改网格。
这是固定 lr/R 的 13 点 eps 筛选，再对保留 eps 局部联合优化，不是全局三维穷举。方案 A 固定 R=1；方案 B 联合搜索 lr/R/eps。
优化器、二阶矩、Momentum-Bias 四频带 MF 和随机流协议沿用 Exp15b。历史训练全部只读，本实验重新训练所有候选。
ε=8、δ=1e-5 仅是单次完整五轮训练预算，无采样放大，不包括搜索及多次训练的组合预算。原始训练诊断仅作私有审计。
十种子样本标准差与同种子配对 Student-t 95% CI 未作多重比较校正。固定超参数差与独立调参差分开报告。

|实验|方案|任务|方法|eps|lr|R|准确率均值 ± 样本SD|交叉熵均值 ± 样本SD|
|---|---|---|---|---|---|---|---|---|
|Exp15d|A|cv|stpex|1.0|2.0|1.0|0.762490 ± 0.002882|0.875508 ± 0.012551|
|Exp15d|A|cv|stpex2|1.0|2.0|1.0|0.762490 ± 0.002882|0.875508 ± 0.012551|
|Exp15d|A|cv|stpex-adam3|1.0|2.0|1.0|0.762470 ± 0.002896|0.875594 ± 0.012555|
|Exp15d|A|nlp|stpex|1.0|0.3|1.0|0.749885 ± 0.003684|1.171417 ± 0.026403|
|Exp15d|A|nlp|stpex2|0.001|0.3|1.0|0.758601 ± 0.003982|2.245819 ± 0.040164|
|Exp15d|A|nlp|stpex-adam3|1e-07|0.3|1.0|0.762959 ± 0.005874|2.656167 ± 0.039790|
|Exp15d|B|cv|stpex|1.0|0.5|3.0|0.775560 ± 0.003648|0.778573 ± 0.009104|
|Exp15d|B|cv|stpex2|1.0|0.5|3.0|0.775560 ± 0.003648|0.778575 ± 0.009104|
|Exp15d|B|cv|stpex-adam3|1.0|0.5|3.0|0.775520 ± 0.003610|0.778771 ± 0.009108|
|Exp15d|B|nlp|stpex|0.001|0.15|20.0|0.780849 ± 0.010586|1.813058 ± 0.104210|
|Exp15d|B|nlp|stpex2|1e-07|0.2|20.0|0.784518 ± 0.009339|1.023964 ± 0.046020|
|Exp15d|B|nlp|stpex-adam3|1e-08|0.2|20.0|0.783257 ± 0.010566|1.497423 ± 0.080536|
|Exp15b|A|cv|momentum|None|2.0|1.0|0.788680 ± 0.002776|0.782865 ± 0.010923|
|Exp15b|A|cv|stpex|0.05|1.0|1.0|0.754110 ± 0.007194|0.864988 ± 0.018380|
|Exp15b|A|cv|stpex2|0.05|1.0|1.0|0.754190 ± 0.007182|0.863959 ± 0.018348|
|Exp15b|A|cv|stpex-adam3|0.05|1.0|1.0|0.754050 ± 0.007275|0.865054 ± 0.018385|
|Exp15b|A|nlp|momentum|None|0.3|1.0|0.750229 ± 0.004829|1.170997 ± 0.020325|
|Exp15b|A|nlp|stpex|0.01|0.3|1.0|0.755734 ± 0.004805|2.092412 ± 0.042408|
|Exp15b|A|nlp|stpex2|0.01|0.3|1.0|0.758142 ± 0.003915|2.026162 ± 0.042124|
|Exp15b|A|nlp|stpex-adam3|0.01|0.3|1.0|0.756078 ± 0.003708|2.074138 ± 0.042492|
|Exp15b|B|cv|momentum|None|0.5|3.0|0.791420 ± 0.003617|0.720972 ± 0.010988|
|Exp15b|B|cv|stpex|0.05|0.5|3.0|0.766810 ± 0.004184|0.861812 ± 0.012505|
|Exp15b|B|cv|stpex2|0.05|0.5|3.0|0.767150 ± 0.004246|0.857726 ± 0.012336|
|Exp15b|B|cv|stpex-adam3|0.05|0.5|3.0|0.766750 ± 0.004221|0.861753 ± 0.012503|
|Exp15b|B|nlp|momentum|None|0.3|20.0|0.786697 ± 0.011404|0.532878 ± 0.034504|
|Exp15b|B|nlp|stpex|0.01|0.15|20.0|0.779472 ± 0.008200|1.346685 ± 0.052880|
|Exp15b|B|nlp|stpex2|0.01|0.2|20.0|0.783716 ± 0.009335|0.988583 ± 0.043171|
|Exp15b|B|nlp|stpex-adam3|0.01|0.15|20.0|0.783142 ± 0.010433|1.253610 ± 0.066826|

## 最佳 eps、相对历史固定 eps 与 Momentum+MF 的改善

以下均为独立调参后的正式十种子差，正值表示右侧较高（交叉熵越低越好）。
A/cv Exp15d/stpex − Exp15b/stpex，accuracy：0.008380，配对 95% CI [0.004546, 0.012214]。
A/cv Exp15d/stpex − Exp15b/stpex，validation_loss：0.010520，配对 95% CI [-0.001147, 0.022187]。
A/cv Exp15d/stpex − Exp15b/momentum，accuracy：-0.026190，配对 95% CI [-0.027885, -0.024495]。
A/cv Exp15d/stpex − Exp15b/momentum，validation_loss：0.092643，配对 95% CI [0.086823, 0.098464]。
A/cv Exp15d/stpex2 − Exp15b/stpex2，accuracy：0.008300，配对 95% CI [0.004461, 0.012139]。
A/cv Exp15d/stpex2 − Exp15b/stpex2，validation_loss：0.011550，配对 95% CI [-0.000140, 0.023239]。
A/cv Exp15d/stpex2 − Exp15b/momentum，accuracy：-0.026190，配对 95% CI [-0.027885, -0.024495]。
A/cv Exp15d/stpex2 − Exp15b/momentum，validation_loss：0.092644，配对 95% CI [0.086823, 0.098464]。
A/cv Exp15d/stpex-adam3 − Exp15b/stpex-adam3，accuracy：0.008420，配对 95% CI [0.004550, 0.012290]。
A/cv Exp15d/stpex-adam3 − Exp15b/stpex-adam3，validation_loss：0.010540，配对 95% CI [-0.001131, 0.022212]。
A/cv Exp15d/stpex-adam3 − Exp15b/momentum，accuracy：-0.026210，配对 95% CI [-0.027900, -0.024520]。
A/cv Exp15d/stpex-adam3 − Exp15b/momentum，validation_loss：0.092730，配对 95% CI [0.086908, 0.098552]。
A/nlp Exp15d/stpex − Exp15b/stpex，accuracy：-0.005849，配对 95% CI [-0.008085, -0.003612]。
A/nlp Exp15d/stpex − Exp15b/stpex，validation_loss：-0.920995，配对 95% CI [-0.943246, -0.898744]。
A/nlp Exp15d/stpex − Exp15b/momentum，accuracy：-0.000344，配对 95% CI [-0.002698, 0.002010]。
A/nlp Exp15d/stpex − Exp15b/momentum，validation_loss：0.000419，配对 95% CI [-0.006497, 0.007335]。
A/nlp Exp15d/stpex2 − Exp15b/stpex2，accuracy：0.000459，配对 95% CI [-0.002362, 0.003279]。
A/nlp Exp15d/stpex2 − Exp15b/stpex2，validation_loss：0.219657，配对 95% CI [0.211529, 0.227784]。
A/nlp Exp15d/stpex2 − Exp15b/momentum，accuracy：0.008372，配对 95% CI [0.005301, 0.011442]。
A/nlp Exp15d/stpex2 − Exp15b/momentum，validation_loss：1.074821，配对 95% CI [1.051633, 1.098009]。
A/nlp Exp15d/stpex-adam3 − Exp15b/stpex-adam3，accuracy：0.006881，配对 95% CI [0.002144, 0.011617]。
A/nlp Exp15d/stpex-adam3 − Exp15b/stpex-adam3，validation_loss：0.582029，配对 95% CI [0.551452, 0.612606]。
A/nlp Exp15d/stpex-adam3 − Exp15b/momentum，accuracy：0.012729，配对 95% CI [0.008665, 0.016794]。
A/nlp Exp15d/stpex-adam3 − Exp15b/momentum，validation_loss：1.485169，配对 95% CI [1.454543, 1.515795]。
B/cv Exp15d/stpex − Exp15b/stpex，accuracy：0.008750，配对 95% CI [0.007094, 0.010406]。
B/cv Exp15d/stpex − Exp15b/stpex，validation_loss：-0.083239，配对 95% CI [-0.087284, -0.079194]。
B/cv Exp15d/stpex − Exp15b/momentum，accuracy：-0.015860，配对 95% CI [-0.017353, -0.014367]。
B/cv Exp15d/stpex − Exp15b/momentum，validation_loss：0.057602，配对 95% CI [0.052805, 0.062399]。
B/cv Exp15d/stpex2 − Exp15b/stpex2，accuracy：0.008410，配对 95% CI [0.006758, 0.010062]。
B/cv Exp15d/stpex2 − Exp15b/stpex2，validation_loss：-0.079151，配对 95% CI [-0.083023, -0.075279]。
B/cv Exp15d/stpex2 − Exp15b/momentum，accuracy：-0.015860，配对 95% CI [-0.017353, -0.014367]。
B/cv Exp15d/stpex2 − Exp15b/momentum，validation_loss：0.057603，配对 95% CI [0.052807, 0.062400]。
B/cv Exp15d/stpex-adam3 − Exp15b/stpex-adam3，accuracy：0.008770，配对 95% CI [0.007148, 0.010392]。
B/cv Exp15d/stpex-adam3 − Exp15b/stpex-adam3，validation_loss：-0.082982，配对 95% CI [-0.087010, -0.078953]。
B/cv Exp15d/stpex-adam3 − Exp15b/momentum，accuracy：-0.015900，配对 95% CI [-0.017356, -0.014444]。
B/cv Exp15d/stpex-adam3 − Exp15b/momentum，validation_loss：0.057800，配对 95% CI [0.053008, 0.062591]。
B/nlp Exp15d/stpex − Exp15b/stpex，accuracy：0.001376，配对 95% CI [-0.002564, 0.005316]。
B/nlp Exp15d/stpex − Exp15b/stpex，validation_loss：0.466373，配对 95% CI [0.419196, 0.513550]。
B/nlp Exp15d/stpex − Exp15b/momentum，accuracy：-0.005849，配对 95% CI [-0.016790, 0.005093]。
B/nlp Exp15d/stpex − Exp15b/momentum，validation_loss：1.280180，配对 95% CI [1.199182, 1.361178]。
B/nlp Exp15d/stpex2 − Exp15b/stpex2，accuracy：0.000803，配对 95% CI [-0.005373, 0.006979]。
B/nlp Exp15d/stpex2 − Exp15b/stpex2，validation_loss：0.035381，配对 95% CI [0.010381, 0.060380]。
B/nlp Exp15d/stpex2 − Exp15b/momentum，accuracy：-0.002179，配对 95% CI [-0.007287, 0.002929]。
B/nlp Exp15d/stpex2 − Exp15b/momentum，validation_loss：0.491086，配对 95% CI [0.455210, 0.526961]。
B/nlp Exp15d/stpex-adam3 − Exp15b/stpex-adam3，accuracy：0.000115，配对 95% CI [-0.003200, 0.003429]。
B/nlp Exp15d/stpex-adam3 − Exp15b/stpex-adam3，validation_loss：0.243812，配对 95% CI [0.206346, 0.281279]。
B/nlp Exp15d/stpex-adam3 − Exp15b/momentum，accuracy：-0.003440，配对 95% CI [-0.010768, 0.003887]。
B/nlp Exp15d/stpex-adam3 − Exp15b/momentum，validation_loss：0.964545，配对 95% CI [0.910540, 1.018549]。
A/cv/Exp15d/stpex 是否超过历史 Momentum+MF：均值=否；配对95%区间下界>0=否。
A/cv/Exp15d/stpex2 是否超过历史 Momentum+MF：均值=否；配对95%区间下界>0=否。
A/cv/Exp15d/stpex-adam3 是否超过历史 Momentum+MF：均值=否；配对95%区间下界>0=否。
A/nlp/Exp15d/stpex 是否超过历史 Momentum+MF：均值=否；配对95%区间下界>0=否。
A/nlp/Exp15d/stpex2 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
A/nlp/Exp15d/stpex-adam3 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
B/cv/Exp15d/stpex 是否超过历史 Momentum+MF：均值=否；配对95%区间下界>0=否。
B/cv/Exp15d/stpex2 是否超过历史 Momentum+MF：均值=否；配对95%区间下界>0=否。
B/cv/Exp15d/stpex-adam3 是否超过历史 Momentum+MF：均值=否；配对95%区间下界>0=否。
B/nlp/Exp15d/stpex 是否超过历史 Momentum+MF：均值=否；配对95%区间下界>0=否。
B/nlp/Exp15d/stpex2 是否超过历史 Momentum+MF：均值=否；配对95%区间下界>0=否。
B/nlp/Exp15d/stpex-adam3 是否超过历史 Momentum+MF：均值=否；配对95%区间下界>0=否。

## 单独改变 eps 的受控效果

controlled_differences.csv 记录同 lr/R、同种子、同训练阶段的 eps−历史 eps 差；搜索/复核是内部验证，不能视为十种子正式单因素证据。
cv/stpex 固定 lr/R 的验证准确率差范围：[-0.023400, 0.009000]（配置范围，非 CI）。
cv/stpex2 固定 lr/R 的验证准确率差范围：[-0.011800, 0.009200]（配置范围，非 CI）。
cv/stpex-adam3 固定 lr/R 的验证准确率差范围：[-0.021200, 0.008800]（配置范围，非 CI）。
nlp/stpex 固定 lr/R 的验证准确率差范围：[-0.055337, 0.002991]（配置范围，非 CI）。
nlp/stpex2 固定 lr/R 的验证准确率差范围：[-0.047486, 0.005422]（配置范围，非 CI）。
nlp/stpex-adam3 固定 lr/R 的验证准确率差范围：[-0.012339, -0.000561]（配置范围，非 CI）。
正式固定 lr/R 对照 A/nlp/Exp15d/stpex，accuracy 差 -0.005849，同种子配对95% CI [-0.008085, -0.003612]。
正式固定 lr/R 对照 A/nlp/Exp15d/stpex，validation_loss 差 -0.920995，同种子配对95% CI [-0.943246, -0.898744]。
正式固定 lr/R 对照 A/nlp/Exp15d/stpex2，accuracy 差 0.000459，同种子配对95% CI [-0.002362, 0.003279]。
正式固定 lr/R 对照 A/nlp/Exp15d/stpex2，validation_loss 差 0.219657，同种子配对95% CI [0.211529, 0.227784]。
正式固定 lr/R 对照 A/nlp/Exp15d/stpex-adam3，accuracy 差 0.006881，同种子配对95% CI [0.002144, 0.011617]。
正式固定 lr/R 对照 A/nlp/Exp15d/stpex-adam3，validation_loss 差 0.582029，同种子配对95% CI [0.551452, 0.612606]。
正式固定 lr/R 对照 B/cv/Exp15d/stpex，accuracy 差 0.008750，同种子配对95% CI [0.007094, 0.010406]。
正式固定 lr/R 对照 B/cv/Exp15d/stpex，validation_loss 差 -0.083239，同种子配对95% CI [-0.087284, -0.079194]。
正式固定 lr/R 对照 B/cv/Exp15d/stpex2，accuracy 差 0.008410，同种子配对95% CI [0.006758, 0.010062]。
正式固定 lr/R 对照 B/cv/Exp15d/stpex2，validation_loss 差 -0.079151，同种子配对95% CI [-0.083023, -0.075279]。
正式固定 lr/R 对照 B/cv/Exp15d/stpex-adam3，accuracy 差 0.008770，同种子配对95% CI [0.007148, 0.010392]。
正式固定 lr/R 对照 B/cv/Exp15d/stpex-adam3，validation_loss 差 -0.082982，同种子配对95% CI [-0.087010, -0.078953]。
正式固定 lr/R 对照 B/nlp/Exp15d/stpex，accuracy 差 0.001376，同种子配对95% CI [-0.002564, 0.005316]。
正式固定 lr/R 对照 B/nlp/Exp15d/stpex，validation_loss 差 0.466373，同种子配对95% CI [0.419196, 0.513550]。
正式固定 lr/R 对照 B/nlp/Exp15d/stpex2，accuracy 差 0.000803，同种子配对95% CI [-0.005373, 0.006979]。
正式固定 lr/R 对照 B/nlp/Exp15d/stpex2，validation_loss 差 0.035381，同种子配对95% CI [0.010381, 0.060380]。

## 联合调整后的排名

A/cv：Exp15b stpex2 > stpex > stpex-adam3；Exp15d stpex > stpex2 > stpex-adam3；变化=True。
A/nlp：Exp15b stpex2 > stpex-adam3 > stpex；Exp15d stpex-adam3 > stpex2 > stpex；变化=True。
B/cv：Exp15b stpex2 > stpex > stpex-adam3；Exp15d stpex > stpex2 > stpex-adam3；变化=True。
B/nlp：Exp15b stpex2 > stpex-adam3 > stpex；Exp15d stpex2 > stpex-adam3 > stpex；变化=False。

## 极小 eps、裁剪与优化稳定性

eps_sensitivity.csv 提供所有搜索候选的裁剪饱和步比例（逐步裁剪比例≥99%）、分母极值、最大更新范数、二阶矩和 non_finite 状态。以下是极小 eps（≤1e-6）的实际诊断，较大更新和饱和不能单独证明因果。
cv/stpex eps=1e-08, lr=0.5, R=3：status=completed, clip=1.0, saturation=1.0, d_min=9.99999993922529e-09, max_update=8.506405192931426。
cv/stpex eps=1e-07, lr=0.5, R=3：status=completed, clip=0.9987911111111112, saturation=0.9955555555555555, d_min=1.0000000116860974e-07, max_update=8.506405192834276。
cv/stpex eps=1e-06, lr=0.5, R=3：status=completed, clip=0.9947555555555555, saturation=0.7688888888888888, d_min=9.999999974752427e-07, max_update=8.506405192820125。
cv/stpex2 eps=1e-08, lr=0.5, R=3：status=completed, clip=0.9670711111111111, saturation=0.47555555555555556, d_min=9.99999993922529e-09, max_update=8.506405192931426。
cv/stpex2 eps=1e-07, lr=0.5, R=3：status=completed, clip=0.9671333333333334, saturation=0.47555555555555556, d_min=1.0000000116860974e-07, max_update=8.506405192834276。
cv/stpex2 eps=1e-06, lr=0.5, R=3：status=completed, clip=0.9668666666666667, saturation=0.4711111111111111, d_min=9.999999974752427e-07, max_update=8.506405192820125。
cv/stpex-adam3 eps=1e-08, lr=0.5, R=3：status=completed, clip=0.9833866666666666, saturation=0.5822222222222222, d_min=9.99999993922529e-09, max_update=8.506405192931426。
cv/stpex-adam3 eps=1e-07, lr=0.5, R=3：status=completed, clip=0.9832933333333334, saturation=0.5822222222222222, d_min=1.0000000116860974e-07, max_update=8.506405192834276。
cv/stpex-adam3 eps=1e-06, lr=0.5, R=3：status=completed, clip=0.9830311111111112, saturation=0.5733333333333334, d_min=9.999999974752427e-07, max_update=8.506405192820125。
nlp/stpex eps=1e-08, lr=0.15, R=20：status=completed, clip=0.6661548387096774, saturation=0.11612903225806452, d_min=9.99999993922529e-09, max_update=15.779641649922544。
nlp/stpex eps=1e-07, lr=0.15, R=20：status=completed, clip=0.43376774193548395, saturation=0.03225806451612903, d_min=1.0000000116860974e-07, max_update=15.779641649596623。
nlp/stpex eps=1e-06, lr=0.15, R=20：status=completed, clip=0.37740645161290326, saturation=0.025806451612903226, d_min=9.999999974752427e-07, max_update=15.779641649479881。
nlp/stpex2 eps=1e-08, lr=0.2, R=20：status=completed, clip=0.34980645161290325, saturation=0.02903225806451613, d_min=9.99999993922529e-09, max_update=21.039521551310063。
nlp/stpex2 eps=1e-07, lr=0.2, R=20：status=completed, clip=0.34693870967741935, saturation=0.02258064516129032, d_min=1.0000000116860974e-07, max_update=21.03952155077348。
nlp/stpex2 eps=1e-06, lr=0.2, R=20：status=completed, clip=0.3458387096774193, saturation=0.02258064516129032, d_min=9.999999974752427e-07, max_update=21.03952155058313。
nlp/stpex-adam3 eps=1e-08, lr=0.15, R=20：status=completed, clip=0.3445096774193548, saturation=0.00967741935483871, d_min=9.99999993922529e-09, max_update=15.779641649922544。
nlp/stpex-adam3 eps=1e-07, lr=0.15, R=20：status=completed, clip=0.34386774193548386, saturation=0.00967741935483871, d_min=1.0000000116860974e-07, max_update=15.779641649596623。
nlp/stpex-adam3 eps=1e-06, lr=0.15, R=20：status=completed, clip=0.3434806451612903, saturation=0.00967741935483871, d_min=9.999999974752427e-07, max_update=15.779641649479881。
nlp/stpex2 eps=1e-07, lr=0.15, R=1：status=completed, clip=0.8327354838709679, saturation=0.1935483870967742, d_min=1.0000000116860974e-07, max_update=0.7889821274171284。
nlp/stpex2 eps=1e-07, lr=0.15, R=10：status=completed, clip=0.40463225806451614, saturation=0.012903225806451613, d_min=1.0000000116860974e-07, max_update=7.889820824798312。
nlp/stpex2 eps=1e-07, lr=0.15, R=20：status=completed, clip=0.34635806451612905, saturation=0.0064516129032258064, d_min=1.0000000116860974e-07, max_update=15.779641649596623。
nlp/stpex2 eps=1e-07, lr=0.2, R=1：status=completed, clip=0.7782870967741936, saturation=0.15161290322580645, d_min=1.0000000116860974e-07, max_update=1.0519761375005203。
nlp/stpex2 eps=1e-07, lr=0.2, R=10：status=completed, clip=0.3746032258064516, saturation=0.00967741935483871, d_min=1.0000000116860974e-07, max_update=10.51976077538674。
nlp/stpex2 eps=1e-07, lr=0.3, R=1：status=completed, clip=0.6922258064516129, saturation=0.1032258064516129, d_min=1.0000000116860974e-07, max_update=1.5779642548342567。
nlp/stpex2 eps=1e-07, lr=0.3, R=10：status=completed, clip=0.3447516129032258, saturation=0.0064516129032258064, d_min=1.0000000116860974e-07, max_update=15.779641649596623。
nlp/stpex2 eps=1e-07, lr=0.3, R=20：status=completed, clip=0.42838709677419357, saturation=0.07741935483870968, d_min=1.0000000116860974e-07, max_update=31.559283299193247。
nlp/stpex-adam3 eps=1e-08, lr=0.15, R=1：status=completed, clip=0.8732322580645161, saturation=0.3419354838709677, d_min=9.99999993922529e-09, max_update=0.7889821274156477。
nlp/stpex-adam3 eps=1e-08, lr=0.15, R=10：status=completed, clip=0.41397096774193554, saturation=0.01935483870967742, d_min=9.99999993922529e-09, max_update=7.889820824961272。
nlp/stpex-adam3 eps=1e-08, lr=0.2, R=1：status=completed, clip=0.8393290322580645, saturation=0.25161290322580643, d_min=9.99999993922529e-09, max_update=1.0519761375043102。
nlp/stpex-adam3 eps=1e-08, lr=0.2, R=10：status=completed, clip=0.38149677419354844, saturation=0.02258064516129032, d_min=9.99999993922529e-09, max_update=10.519760775655032。
nlp/stpex-adam3 eps=1e-08, lr=0.2, R=20：status=completed, clip=0.3387387096774193, saturation=0.02258064516129032, d_min=9.99999993922529e-09, max_update=21.039521551310063。
nlp/stpex-adam3 eps=1e-08, lr=0.3, R=1：status=completed, clip=0.7333193548387097, saturation=0.16774193548387098, d_min=9.99999993922529e-09, max_update=1.5779642548312953。
nlp/stpex-adam3 eps=1e-08, lr=0.3, R=10：status=completed, clip=0.34565483870967745, saturation=0.00967741935483871, d_min=9.99999993922529e-09, max_update=15.779641649922544。
nlp/stpex-adam3 eps=1e-08, lr=0.3, R=20：status=completed, clip=0.3670548387096774, saturation=0.03225806451612903, d_min=9.99999993922529e-09, max_update=31.559283299845088。
nlp/stpex-adam3 eps=1e-07, lr=0.15, R=1：status=completed, clip=0.8728806451612902, saturation=0.3419354838709677, d_min=1.0000000116860974e-07, max_update=0.7889821274171284。
nlp/stpex-adam3 eps=1e-07, lr=0.15, R=10：status=completed, clip=0.41407096774193547, saturation=0.01935483870967742, d_min=1.0000000116860974e-07, max_update=7.889820824798312。
nlp/stpex-adam3 eps=1e-07, lr=0.2, R=1：status=completed, clip=0.839009677419355, saturation=0.25161290322580643, d_min=1.0000000116860974e-07, max_update=1.0519761375005203。
nlp/stpex-adam3 eps=1e-07, lr=0.2, R=10：status=completed, clip=0.38077096774193553, saturation=0.02258064516129032, d_min=1.0000000116860974e-07, max_update=10.51976077538674。
nlp/stpex-adam3 eps=1e-07, lr=0.2, R=20：status=completed, clip=0.3371258064516129, saturation=0.02258064516129032, d_min=1.0000000116860974e-07, max_update=21.03952155077348。
nlp/stpex-adam3 eps=1e-07, lr=0.3, R=1：status=completed, clip=0.7332709677419356, saturation=0.16774193548387098, d_min=1.0000000116860974e-07, max_update=1.5779642548342567。
nlp/stpex-adam3 eps=1e-07, lr=0.3, R=10：status=completed, clip=0.34521612903225807, saturation=0.00967741935483871, d_min=1.0000000116860974e-07, max_update=15.779641649596623。
nlp/stpex-adam3 eps=1e-07, lr=0.3, R=20：status=completed, clip=0.4454064516129032, saturation=0.10967741935483871, d_min=1.0000000116860974e-07, max_update=31.559283299193247。

固定 lr/R 和联合调参的 eps—准确率、交叉熵、裁剪比例曲线见 curves/。联合曲线按每个保留 eps 的单种子验证最优 lr/R 展示，正式 eps 由额外两种子复核选定。
搜索 195，复核 48，正式去重 120；非有限搜索/复核候选 0。
