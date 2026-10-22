# Exp15d 最终报告

历史官方评价集已使用过，本次不是新盲测。选参仅使用内部验证；未根据官方结果修改网格。
这是固定 lr/R 的 13 点 eps 筛选，再对保留 eps 局部联合优化，不是全局三维穷举。方案 A 固定 R=1；方案 B 联合搜索 lr/R/eps。
优化器、二阶矩、Momentum-Bias 四频带 MF 和随机流协议沿用 Exp15b。历史训练全部只读，本实验重新训练所有候选。
ε=8、δ=1e-5 仅是单次完整五轮训练预算，无采样放大，不包括搜索及多次训练的组合预算。原始训练诊断仅作私有审计。
十种子样本标准差与同种子配对 Student-t 95% CI 未作多重比较校正。固定超参数差与独立调参差分开报告。

|实验|方案|任务|方法|eps|lr|R|准确率均值 ± 样本SD|交叉熵均值 ± 样本SD|
|---|---|---|---|---|---|---|---|---|
|Exp15d|A|cv|stpex|0.3|0.5|1.0|0.701100 ± 0.000876|0.298900 ± 0.000876|
|Exp15d|A|cv|stpex2|0.3|0.5|1.0|0.701100 ± 0.000876|0.298900 ± 0.000876|
|Exp15d|A|cv|stpex-adam3|0.3|0.5|1.0|0.701100 ± 0.000876|0.298900 ± 0.000876|
|Exp15d|A|nlp|stpex|0.3|0.15|1.0|0.701100 ± 0.000876|0.298900 ± 0.000876|
|Exp15d|A|nlp|stpex2|0.3|0.15|1.0|0.701100 ± 0.000876|0.298900 ± 0.000876|
|Exp15d|A|nlp|stpex-adam3|0.3|0.15|1.0|0.701100 ± 0.000876|0.298900 ± 0.000876|
|Exp15d|B|cv|stpex|0.3|0.5|1.0|0.701100 ± 0.000876|0.298900 ± 0.000876|
|Exp15d|B|cv|stpex2|0.3|0.5|1.0|0.701100 ± 0.000876|0.298900 ± 0.000876|
|Exp15d|B|cv|stpex-adam3|0.3|0.5|1.0|0.701100 ± 0.000876|0.298900 ± 0.000876|
|Exp15d|B|nlp|stpex|0.3|0.15|1.0|0.701100 ± 0.000876|0.298900 ± 0.000876|
|Exp15d|B|nlp|stpex2|0.3|0.15|1.0|0.701100 ± 0.000876|0.298900 ± 0.000876|
|Exp15d|B|nlp|stpex-adam3|0.3|0.15|1.0|0.701100 ± 0.000876|0.298900 ± 0.000876|
|Exp15b|A|cv|stpex|0.05|0.5|1.0|0.600000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|A|cv|stpex2|0.05|0.5|1.0|0.600000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|A|cv|stpex-adam3|0.05|0.5|1.0|0.600000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|A|cv|momentum|None|0.5|1.0|0.650000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|A|nlp|stpex|0.01|0.15|1.0|0.600000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|A|nlp|stpex2|0.01|0.15|1.0|0.600000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|A|nlp|stpex-adam3|0.01|0.15|1.0|0.600000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|A|nlp|momentum|None|0.15|1.0|0.650000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|B|cv|stpex|0.05|0.5|1.0|0.600000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|B|cv|stpex2|0.05|0.5|1.0|0.600000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|B|cv|stpex-adam3|0.05|0.5|1.0|0.600000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|B|cv|momentum|None|0.5|1.0|0.650000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|B|nlp|stpex|0.01|0.15|1.0|0.600000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|B|nlp|stpex2|0.01|0.15|1.0|0.600000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|B|nlp|stpex-adam3|0.01|0.15|1.0|0.600000 ± 0.000000|0.400000 ± 0.000000|
|Exp15b|B|nlp|momentum|None|0.15|1.0|0.650000 ± 0.000000|0.400000 ± 0.000000|

## 最佳 eps、相对历史固定 eps 与 Momentum+MF 的改善

以下均为独立调参后的正式十种子差，正值表示右侧较高（交叉熵越低越好）。
A/cv Exp15d/stpex − Exp15b/stpex，accuracy：0.101100，配对 95% CI [0.100474, 0.101726]。
A/cv Exp15d/stpex − Exp15b/stpex，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
A/cv Exp15d/stpex − Exp15b/momentum，accuracy：0.051100，配对 95% CI [0.050474, 0.051726]。
A/cv Exp15d/stpex − Exp15b/momentum，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
A/cv Exp15d/stpex2 − Exp15b/stpex2，accuracy：0.101100，配对 95% CI [0.100474, 0.101726]。
A/cv Exp15d/stpex2 − Exp15b/stpex2，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
A/cv Exp15d/stpex2 − Exp15b/momentum，accuracy：0.051100，配对 95% CI [0.050474, 0.051726]。
A/cv Exp15d/stpex2 − Exp15b/momentum，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
A/cv Exp15d/stpex-adam3 − Exp15b/stpex-adam3，accuracy：0.101100，配对 95% CI [0.100474, 0.101726]。
A/cv Exp15d/stpex-adam3 − Exp15b/stpex-adam3，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
A/cv Exp15d/stpex-adam3 − Exp15b/momentum，accuracy：0.051100，配对 95% CI [0.050474, 0.051726]。
A/cv Exp15d/stpex-adam3 − Exp15b/momentum，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
A/nlp Exp15d/stpex − Exp15b/stpex，accuracy：0.101100，配对 95% CI [0.100474, 0.101726]。
A/nlp Exp15d/stpex − Exp15b/stpex，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
A/nlp Exp15d/stpex − Exp15b/momentum，accuracy：0.051100，配对 95% CI [0.050474, 0.051726]。
A/nlp Exp15d/stpex − Exp15b/momentum，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
A/nlp Exp15d/stpex2 − Exp15b/stpex2，accuracy：0.101100，配对 95% CI [0.100474, 0.101726]。
A/nlp Exp15d/stpex2 − Exp15b/stpex2，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
A/nlp Exp15d/stpex2 − Exp15b/momentum，accuracy：0.051100，配对 95% CI [0.050474, 0.051726]。
A/nlp Exp15d/stpex2 − Exp15b/momentum，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
A/nlp Exp15d/stpex-adam3 − Exp15b/stpex-adam3，accuracy：0.101100，配对 95% CI [0.100474, 0.101726]。
A/nlp Exp15d/stpex-adam3 − Exp15b/stpex-adam3，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
A/nlp Exp15d/stpex-adam3 − Exp15b/momentum，accuracy：0.051100，配对 95% CI [0.050474, 0.051726]。
A/nlp Exp15d/stpex-adam3 − Exp15b/momentum，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
B/cv Exp15d/stpex − Exp15b/stpex，accuracy：0.101100，配对 95% CI [0.100474, 0.101726]。
B/cv Exp15d/stpex − Exp15b/stpex，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
B/cv Exp15d/stpex − Exp15b/momentum，accuracy：0.051100，配对 95% CI [0.050474, 0.051726]。
B/cv Exp15d/stpex − Exp15b/momentum，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
B/cv Exp15d/stpex2 − Exp15b/stpex2，accuracy：0.101100，配对 95% CI [0.100474, 0.101726]。
B/cv Exp15d/stpex2 − Exp15b/stpex2，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
B/cv Exp15d/stpex2 − Exp15b/momentum，accuracy：0.051100，配对 95% CI [0.050474, 0.051726]。
B/cv Exp15d/stpex2 − Exp15b/momentum，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
B/cv Exp15d/stpex-adam3 − Exp15b/stpex-adam3，accuracy：0.101100，配对 95% CI [0.100474, 0.101726]。
B/cv Exp15d/stpex-adam3 − Exp15b/stpex-adam3，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
B/cv Exp15d/stpex-adam3 − Exp15b/momentum，accuracy：0.051100，配对 95% CI [0.050474, 0.051726]。
B/cv Exp15d/stpex-adam3 − Exp15b/momentum，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
B/nlp Exp15d/stpex − Exp15b/stpex，accuracy：0.101100，配对 95% CI [0.100474, 0.101726]。
B/nlp Exp15d/stpex − Exp15b/stpex，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
B/nlp Exp15d/stpex − Exp15b/momentum，accuracy：0.051100，配对 95% CI [0.050474, 0.051726]。
B/nlp Exp15d/stpex − Exp15b/momentum，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
B/nlp Exp15d/stpex2 − Exp15b/stpex2，accuracy：0.101100，配对 95% CI [0.100474, 0.101726]。
B/nlp Exp15d/stpex2 − Exp15b/stpex2，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
B/nlp Exp15d/stpex2 − Exp15b/momentum，accuracy：0.051100，配对 95% CI [0.050474, 0.051726]。
B/nlp Exp15d/stpex2 − Exp15b/momentum，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
B/nlp Exp15d/stpex-adam3 − Exp15b/stpex-adam3，accuracy：0.101100，配对 95% CI [0.100474, 0.101726]。
B/nlp Exp15d/stpex-adam3 − Exp15b/stpex-adam3，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
B/nlp Exp15d/stpex-adam3 − Exp15b/momentum，accuracy：0.051100，配对 95% CI [0.050474, 0.051726]。
B/nlp Exp15d/stpex-adam3 − Exp15b/momentum，validation_loss：-0.101100，配对 95% CI [-0.101726, -0.100474]。
A/cv/Exp15d/stpex 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
A/cv/Exp15d/stpex2 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
A/cv/Exp15d/stpex-adam3 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
A/nlp/Exp15d/stpex 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
A/nlp/Exp15d/stpex2 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
A/nlp/Exp15d/stpex-adam3 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
B/cv/Exp15d/stpex 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
B/cv/Exp15d/stpex2 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
B/cv/Exp15d/stpex-adam3 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
B/nlp/Exp15d/stpex 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
B/nlp/Exp15d/stpex2 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。
B/nlp/Exp15d/stpex-adam3 是否超过历史 Momentum+MF：均值=是；配对95%区间下界>0=是。

## 单独改变 eps 的受控效果

controlled_differences.csv 记录同 lr/R、同种子、同训练阶段的 eps−历史 eps 差；搜索/复核是内部验证，不能视为十种子正式单因素证据。
cv/stpex 固定 lr/R 的验证准确率差范围：[0.000000, 0.000000]（配置范围，非 CI）。
cv/stpex2 固定 lr/R 的验证准确率差范围：[0.000000, 0.000000]（配置范围，非 CI）。
cv/stpex-adam3 固定 lr/R 的验证准确率差范围：[0.000000, 0.000000]（配置范围，非 CI）。
nlp/stpex 固定 lr/R 的验证准确率差范围：[0.000000, 0.000000]（配置范围，非 CI）。
nlp/stpex2 固定 lr/R 的验证准确率差范围：[0.000000, 0.000000]（配置范围，非 CI）。
nlp/stpex-adam3 固定 lr/R 的验证准确率差范围：[0.000000, 0.000000]（配置范围，非 CI）。
正式固定 lr/R 对照 A/cv/Exp15d/stpex，accuracy 差 0.101100，同种子配对95% CI [0.100474, 0.101726]。
正式固定 lr/R 对照 A/cv/Exp15d/stpex，validation_loss 差 -0.101100，同种子配对95% CI [-0.101726, -0.100474]。
正式固定 lr/R 对照 A/cv/Exp15d/stpex2，accuracy 差 0.101100，同种子配对95% CI [0.100474, 0.101726]。
正式固定 lr/R 对照 A/cv/Exp15d/stpex2，validation_loss 差 -0.101100，同种子配对95% CI [-0.101726, -0.100474]。
正式固定 lr/R 对照 A/cv/Exp15d/stpex-adam3，accuracy 差 0.101100，同种子配对95% CI [0.100474, 0.101726]。
正式固定 lr/R 对照 A/cv/Exp15d/stpex-adam3，validation_loss 差 -0.101100，同种子配对95% CI [-0.101726, -0.100474]。
正式固定 lr/R 对照 A/nlp/Exp15d/stpex，accuracy 差 0.101100，同种子配对95% CI [0.100474, 0.101726]。
正式固定 lr/R 对照 A/nlp/Exp15d/stpex，validation_loss 差 -0.101100，同种子配对95% CI [-0.101726, -0.100474]。
正式固定 lr/R 对照 A/nlp/Exp15d/stpex2，accuracy 差 0.101100，同种子配对95% CI [0.100474, 0.101726]。
正式固定 lr/R 对照 A/nlp/Exp15d/stpex2，validation_loss 差 -0.101100，同种子配对95% CI [-0.101726, -0.100474]。
正式固定 lr/R 对照 A/nlp/Exp15d/stpex-adam3，accuracy 差 0.101100，同种子配对95% CI [0.100474, 0.101726]。
正式固定 lr/R 对照 A/nlp/Exp15d/stpex-adam3，validation_loss 差 -0.101100，同种子配对95% CI [-0.101726, -0.100474]。
正式固定 lr/R 对照 B/cv/Exp15d/stpex，accuracy 差 0.101100，同种子配对95% CI [0.100474, 0.101726]。
正式固定 lr/R 对照 B/cv/Exp15d/stpex，validation_loss 差 -0.101100，同种子配对95% CI [-0.101726, -0.100474]。
正式固定 lr/R 对照 B/cv/Exp15d/stpex2，accuracy 差 0.101100，同种子配对95% CI [0.100474, 0.101726]。
正式固定 lr/R 对照 B/cv/Exp15d/stpex2，validation_loss 差 -0.101100，同种子配对95% CI [-0.101726, -0.100474]。
正式固定 lr/R 对照 B/cv/Exp15d/stpex-adam3，accuracy 差 0.101100，同种子配对95% CI [0.100474, 0.101726]。
正式固定 lr/R 对照 B/cv/Exp15d/stpex-adam3，validation_loss 差 -0.101100，同种子配对95% CI [-0.101726, -0.100474]。
正式固定 lr/R 对照 B/nlp/Exp15d/stpex，accuracy 差 0.101100，同种子配对95% CI [0.100474, 0.101726]。
正式固定 lr/R 对照 B/nlp/Exp15d/stpex，validation_loss 差 -0.101100，同种子配对95% CI [-0.101726, -0.100474]。
正式固定 lr/R 对照 B/nlp/Exp15d/stpex2，accuracy 差 0.101100，同种子配对95% CI [0.100474, 0.101726]。
正式固定 lr/R 对照 B/nlp/Exp15d/stpex2，validation_loss 差 -0.101100，同种子配对95% CI [-0.101726, -0.100474]。
正式固定 lr/R 对照 B/nlp/Exp15d/stpex-adam3，accuracy 差 0.101100，同种子配对95% CI [0.100474, 0.101726]。
正式固定 lr/R 对照 B/nlp/Exp15d/stpex-adam3，validation_loss 差 -0.101100，同种子配对95% CI [-0.101726, -0.100474]。

## 联合调整后的排名

A/cv：Exp15b stpex > stpex2 > stpex-adam3；Exp15d stpex > stpex2 > stpex-adam3；变化=False。
A/nlp：Exp15b stpex > stpex2 > stpex-adam3；Exp15d stpex > stpex2 > stpex-adam3；变化=False。
B/cv：Exp15b stpex > stpex2 > stpex-adam3；Exp15d stpex > stpex2 > stpex-adam3；变化=False。
B/nlp：Exp15b stpex > stpex2 > stpex-adam3；Exp15d stpex > stpex2 > stpex-adam3；变化=False。

## 极小 eps、裁剪与优化稳定性

eps_sensitivity.csv 提供所有搜索候选的裁剪饱和步比例（逐步裁剪比例≥99%）、分母极值、最大更新范数、二阶矩和 non_finite 状态。以下是极小 eps（≤1e-6）的实际诊断，较大更新和饱和不能单独证明因果。

固定 lr/R 和联合调参的 eps—准确率、交叉熵、裁剪比例曲线见 curves/。联合曲线按每个保留 eps 的单种子验证最优 lr/R 展示，正式 eps 由额外两种子复核选定。
搜索 195，复核 0，正式去重 60；非有限搜索/复核候选 0。
