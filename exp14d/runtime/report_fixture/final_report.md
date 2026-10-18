# Exp14d 最终报告

所有结果已完成；统计使用 10 个固定种子、样本标准差（ddof=1）和双侧 Student-t 95% 置信区间。配对差按同一种子计算，单位为准确率（0–1）。区间未作多重比较校正。

隐私 ε=8、δ=1e-5 仅针对单次训练；不宣称搜索、选择和重复训练合成后仍为 ε=8。历史官方评价集已经使用过，本实验不是新的盲测。所有选参只使用最后一轮内部验证准确率，依次以验证损失和固定 trial ID 打破并列。

普通 C=1 裁剪约束欧氏梯度范数；Scale 的 C=1 约束上一时刻 vhat 所定义的缩放梯度范数，噪声在该空间加入后逆缩放。两者几何不等价，因此不能把 Scale 差异归结为相同欧氏阈值下的收益。CV/NLP eps_scale 分别固定为 0.05/0.01。

训练严格复用 Exp9/Exp14c：CV 搜索 45,000 条、正式 50,000 条；NLP 搜索和正式均为历史固定的 62,000 条训练子集（其余 5,349 条为内部验证），每轮 62 步，并非把全部 67,349 条原始样本重新纳入训练。

## A：学习率边界补充


边界搜索仅扩展预先指定的八个候选，不证明连续学习率空间的全局最优。合并排名及来源哈希见 A_search_ranking.csv、A_frozen_manifest.json。

## 六种算法：原始配置、边界更新、C=1

|任务|方法|配置|lr|C|均值|样本标准差|95% CI|
|---|---|---|---|---|---|---|---|
|cv|IID + Adam|original|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-SGD + Scale + Adam|original|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-SGD + Adam|original|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-SGD + AdaM³|original|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-Momentum + AdaM³|original|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-Momentum-Bias + AdaM³|original|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|IID + Adam|original|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-SGD + Scale + Adam|original|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-SGD + Adam|original|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-SGD + AdaM³|original|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-Momentum + AdaM³|original|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-Momentum-Bias + AdaM³|original|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|IID + Adam|boundary_updated|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-SGD + Scale + Adam|boundary_updated|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-SGD + Adam|boundary_updated|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-SGD + AdaM³|boundary_updated|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-Momentum + AdaM³|boundary_updated|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-Momentum-Bias + AdaM³|boundary_updated|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|IID + Adam|boundary_updated|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-SGD + Scale + Adam|boundary_updated|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-SGD + Adam|boundary_updated|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-SGD + AdaM³|boundary_updated|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-Momentum + AdaM³|boundary_updated|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-Momentum-Bias + AdaM³|boundary_updated|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|IID + Adam|C1|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-SGD + Scale + Adam|C1|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-SGD + Adam|C1|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-SGD + AdaM³|C1|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-Momentum + AdaM³|C1|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|cv|MF-Momentum-Bias + AdaM³|C1|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|IID + Adam|C1|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-SGD + Scale + Adam|C1|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-SGD + Adam|C1|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-SGD + AdaM³|C1|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-Momentum + AdaM³|C1|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|
|nlp|MF-Momentum-Bias + AdaM³|C1|0.001|1|0.50450|0.00303|[0.50233, 0.50667]|

## 配对差与相对收益

- cv MF-SGD + AdaM³，boundary_updated minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-Momentum + AdaM³，boundary_updated minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-Momentum-Bias + AdaM³，boundary_updated minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-SGD + AdaM³，boundary_updated minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-Momentum + AdaM³，boundary_updated minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-Momentum-Bias + AdaM³，boundary_updated minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv IID + Adam，C1 minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-SGD + Scale + Adam，C1 minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-SGD + Adam，C1 minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-SGD + AdaM³，C1 minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-Momentum + AdaM³，C1 minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-Momentum-Bias + AdaM³，C1 minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp IID + Adam，C1 minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-SGD + Scale + Adam，C1 minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-SGD + Adam，C1 minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-SGD + AdaM³，C1 minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-Momentum + AdaM³，C1 minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-Momentum-Bias + AdaM³，C1 minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv IID + Adam，C1 minus boundary_updated: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-SGD + Scale + Adam，C1 minus boundary_updated: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-SGD + Adam，C1 minus boundary_updated: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-SGD + AdaM³，C1 minus boundary_updated: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-Momentum + AdaM³，C1 minus boundary_updated: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-Momentum-Bias + AdaM³，C1 minus boundary_updated: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp IID + Adam，C1 minus boundary_updated: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-SGD + Scale + Adam，C1 minus boundary_updated: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-SGD + Adam，C1 minus boundary_updated: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-SGD + AdaM³，C1 minus boundary_updated: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-Momentum + AdaM³，C1 minus boundary_updated: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-Momentum-Bias + AdaM³，C1 minus boundary_updated: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，original: MF-SGD + Adam minus IID + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，original: MF-SGD + Scale + Adam minus MF-SGD + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，original: MF-SGD + AdaM³ minus MF-SGD + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，original: MF-Momentum + AdaM³ minus MF-SGD + AdaM³: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，original: MF-Momentum-Bias + AdaM³ minus MF-Momentum + AdaM³: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，original: MF-SGD + Adam minus IID + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，original: MF-SGD + Scale + Adam minus MF-SGD + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，original: MF-SGD + AdaM³ minus MF-SGD + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，original: MF-Momentum + AdaM³ minus MF-SGD + AdaM³: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，original: MF-Momentum-Bias + AdaM³ minus MF-Momentum + AdaM³: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，boundary_updated: MF-SGD + Adam minus IID + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，boundary_updated: MF-SGD + Scale + Adam minus MF-SGD + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，boundary_updated: MF-SGD + AdaM³ minus MF-SGD + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，boundary_updated: MF-Momentum + AdaM³ minus MF-SGD + AdaM³: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，boundary_updated: MF-Momentum-Bias + AdaM³ minus MF-Momentum + AdaM³: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，boundary_updated: MF-SGD + Adam minus IID + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，boundary_updated: MF-SGD + Scale + Adam minus MF-SGD + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，boundary_updated: MF-SGD + AdaM³ minus MF-SGD + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，boundary_updated: MF-Momentum + AdaM³ minus MF-SGD + AdaM³: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，boundary_updated: MF-Momentum-Bias + AdaM³ minus MF-Momentum + AdaM³: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，C1: MF-SGD + Adam minus IID + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，C1: MF-SGD + Scale + Adam minus MF-SGD + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，C1: MF-SGD + AdaM³ minus MF-SGD + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，C1: MF-Momentum + AdaM³ minus MF-SGD + AdaM³: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv method comparison，C1: MF-Momentum-Bias + AdaM³ minus MF-Momentum + AdaM³: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，C1: MF-SGD + Adam minus IID + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，C1: MF-SGD + Scale + Adam minus MF-SGD + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，C1: MF-SGD + AdaM³ minus MF-SGD + Adam: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，C1: MF-Momentum + AdaM³ minus MF-SGD + AdaM³: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp method comparison，C1: MF-Momentum-Bias + AdaM³ minus MF-Momentum + AdaM³: +0.00000 [+0.00000, +0.00000]（区间包含零）。

C=1 对照同时改变裁剪阈值及独立选择的学习率；算法对照同样使用各自最优学习率，因此属于完整训练配置比较，不能单独解释为某个机制的因果效应。MF、Scale、AdaM³ 的相对表现以上述配对差为依据。

诊断曲线见 figures/，裁剪比例来自每一步；更新范数、二阶矩和学习率记录前 20 步及之后每 10 步。历史未记录的优化器诊断明确标为缺失，不重跑补齐。所有新增训练保留检查点、训练曲线和噪声校准记录。

来源与审计：historical_sources_sha256.json、report_sources_sha256.json、source_provenance.json、scheduler.jsonl、failures.jsonl、final_audit.json。
