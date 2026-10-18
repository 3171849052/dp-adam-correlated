# Exp14d 最终报告

所有结果已完成；统计使用 10 个固定种子、样本标准差（ddof=1）和双侧 Student-t 95% 置信区间。配对差按同一种子计算，单位为准确率（0–1）。区间未作多重比较校正。

隐私 ε=8、δ=1e-5 仅针对单次训练；不宣称搜索、选择和重复训练合成后仍为 ε=8。历史官方评价集已经使用过，本实验不是新的盲测。所有选参只使用最后一轮内部验证准确率，依次以验证损失和固定 trial ID 打破并列。

普通 C=1 裁剪约束欧氏梯度范数；Scale 的 C=1 约束上一时刻 vhat 所定义的缩放梯度范数，噪声在该空间加入后逆缩放。两者几何不等价，因此不能把 Scale 差异归结为相同欧氏阈值下的收益。CV/NLP eps_scale 分别固定为 0.05/0.01。

训练严格复用 Exp9/Exp14c：CV 搜索 45,000 条、正式 50,000 条；NLP 搜索和正式均为历史固定的 62,000 条训练子集（其余 5,349 条为内部验证），每轮 62 步，并非把全部 67,349 条原始样本重新纳入训练。

## A：学习率边界补充

- cv dp-adam-momentum-bandinvmf-adam3: (0.0006, C=30) → (0.0006, C=30); 引用 Exp14c 十种子结果。
- cv dp-adam-momentum-bias-bandinvmf-adam3: (0.0004, C=30) → (0.0004, C=30); 引用 Exp14c 十种子结果。
- cv dp-adam-sgd-bandinvmf-adam3: (0.0002, C=30) → (0.0002, C=30); 引用 Exp14c 十种子结果。
- nlp dp-adam-momentum-bandinvmf-adam3: (0.001, C=10) → (0.0015, C=10); 重跑全部十种子。
- nlp dp-adam-momentum-bias-bandinvmf-adam3: (0.001, C=20) → (0.0015, C=20); 重跑全部十种子。
- nlp dp-adam-sgd-bandinvmf-adam3: (0.0008, C=10) → (0.0008, C=10); 引用 Exp14c 十种子结果。

边界搜索仅扩展预先指定的八个候选，不证明连续学习率空间的全局最优。合并排名及来源哈希见 A_search_ranking.csv、A_frozen_manifest.json。

## 六种算法：原始配置、边界更新、C=1

|任务|方法|配置|lr|C|均值|样本标准差|95% CI|
|---|---|---|---|---|---|---|---|
|cv|IID + Adam|original|0.0005|30|0.60189|0.01796|[0.58904, 0.61474]|
|cv|MF-SGD + Scale + Adam|original|0.003|200|0.70610|0.00701|[0.70109, 0.71111]|
|cv|MF-SGD + Adam|original|0.003|30|0.70223|0.00531|[0.69843, 0.70603]|
|cv|MF-SGD + AdaM³|original|0.0002|30|0.73057|0.00750|[0.72520, 0.73594]|
|cv|MF-Momentum + AdaM³|original|0.0006|30|0.76411|0.00610|[0.75974, 0.76848]|
|cv|MF-Momentum-Bias + AdaM³|original|0.0004|30|0.77082|0.00434|[0.76772, 0.77392]|
|nlp|IID + Adam|original|0.002|0.3|0.75986|0.00911|[0.75334, 0.76638]|
|nlp|MF-SGD + Scale + Adam|original|0.005|300|0.78647|0.00635|[0.78193, 0.79101]|
|nlp|MF-SGD + Adam|original|0.005|20|0.77718|0.00793|[0.77151, 0.78285]|
|nlp|MF-SGD + AdaM³|original|0.0008|10|0.77718|0.01022|[0.76987, 0.78449]|
|nlp|MF-Momentum + AdaM³|original|0.001|10|0.77970|0.00910|[0.77319, 0.78621]|
|nlp|MF-Momentum-Bias + AdaM³|original|0.001|20|0.78601|0.01203|[0.77740, 0.79461]|
|cv|IID + Adam|boundary_updated|0.0005|30|0.60189|0.01796|[0.58904, 0.61474]|
|cv|MF-SGD + Scale + Adam|boundary_updated|0.003|200|0.70610|0.00701|[0.70109, 0.71111]|
|cv|MF-SGD + Adam|boundary_updated|0.003|30|0.70223|0.00531|[0.69843, 0.70603]|
|cv|MF-SGD + AdaM³|boundary_updated|0.0002|30|0.73057|0.00750|[0.72520, 0.73594]|
|cv|MF-Momentum + AdaM³|boundary_updated|0.0006|30|0.76411|0.00610|[0.75974, 0.76848]|
|cv|MF-Momentum-Bias + AdaM³|boundary_updated|0.0004|30|0.77082|0.00434|[0.76772, 0.77392]|
|nlp|IID + Adam|boundary_updated|0.002|0.3|0.75986|0.00911|[0.75334, 0.76638]|
|nlp|MF-SGD + Scale + Adam|boundary_updated|0.005|300|0.78647|0.00635|[0.78193, 0.79101]|
|nlp|MF-SGD + Adam|boundary_updated|0.005|20|0.77718|0.00793|[0.77151, 0.78285]|
|nlp|MF-SGD + AdaM³|boundary_updated|0.0008|10|0.77718|0.01022|[0.76987, 0.78449]|
|nlp|MF-Momentum + AdaM³|boundary_updated|0.0015|10|0.78016|0.01076|[0.77246, 0.78786]|
|nlp|MF-Momentum-Bias + AdaM³|boundary_updated|0.0015|20|0.78589|0.01157|[0.77762, 0.79417]|
|cv|IID + Adam|C1|0.0005|1|0.59428|0.01715|[0.58201, 0.60655]|
|cv|MF-SGD + Scale + Adam|C1|0.003|1|0.69284|0.00688|[0.68792, 0.69776]|
|cv|MF-SGD + Adam|C1|0.003|1|0.70029|0.00635|[0.69575, 0.70483]|
|cv|MF-SGD + AdaM³|C1|0.0002|1|0.72127|0.00827|[0.71535, 0.72719]|
|cv|MF-Momentum + AdaM³|C1|0.0006|1|0.75660|0.00725|[0.75142, 0.76178]|
|cv|MF-Momentum-Bias + AdaM³|C1|0.0004|1|0.76423|0.00542|[0.76035, 0.76811]|
|nlp|IID + Adam|C1|0.002|1|0.75894|0.00841|[0.75293, 0.76496]|
|nlp|MF-SGD + Scale + Adam|C1|0.005|1|0.78131|0.01030|[0.77394, 0.78868]|
|nlp|MF-SGD + Adam|C1|0.003|1|0.77856|0.00956|[0.77172, 0.78539]|
|nlp|MF-SGD + AdaM³|C1|0.0008|1|0.78005|0.00807|[0.77427, 0.78582]|
|nlp|MF-Momentum + AdaM³|C1|0.0015|1|0.78257|0.01057|[0.77501, 0.79013]|
|nlp|MF-Momentum-Bias + AdaM³|C1|0.0015|1|0.78475|0.01014|[0.77749, 0.79200]|

## 配对差与相对收益

- cv MF-SGD + AdaM³，boundary_updated minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-Momentum + AdaM³，boundary_updated minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- cv MF-Momentum-Bias + AdaM³，boundary_updated minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-SGD + AdaM³，boundary_updated minus original: +0.00000 [+0.00000, +0.00000]（区间包含零）。
- nlp MF-Momentum + AdaM³，boundary_updated minus original: +0.00046 [-0.00624, +0.00716]（区间包含零）。
- nlp MF-Momentum-Bias + AdaM³，boundary_updated minus original: -0.00011 [-0.00682, +0.00659]（区间包含零）。
- cv IID + Adam，C1 minus original: -0.00761 [-0.00915, -0.00607]（负向）。
- cv MF-SGD + Scale + Adam，C1 minus original: -0.01326 [-0.01548, -0.01104]（负向）。
- cv MF-SGD + Adam，C1 minus original: -0.00194 [-0.00328, -0.00060]（负向）。
- cv MF-SGD + AdaM³，C1 minus original: -0.00930 [-0.01127, -0.00733]（负向）。
- cv MF-Momentum + AdaM³，C1 minus original: -0.00751 [-0.00940, -0.00562]（负向）。
- cv MF-Momentum-Bias + AdaM³，C1 minus original: -0.00659 [-0.00793, -0.00525]（负向）。
- nlp IID + Adam，C1 minus original: -0.00092 [-0.00259, +0.00076]（区间包含零）。
- nlp MF-SGD + Scale + Adam，C1 minus original: -0.00516 [-0.01173, +0.00141]（区间包含零）。
- nlp MF-SGD + Adam，C1 minus original: +0.00138 [-0.00517, +0.00793]（区间包含零）。
- nlp MF-SGD + AdaM³，C1 minus original: +0.00287 [-0.00143, +0.00716]（区间包含零）。
- nlp MF-Momentum + AdaM³，C1 minus original: +0.00287 [-0.00493, +0.01066]（区间包含零）。
- nlp MF-Momentum-Bias + AdaM³，C1 minus original: -0.00126 [-0.00868, +0.00616]（区间包含零）。
- cv IID + Adam，C1 minus boundary_updated: -0.00761 [-0.00915, -0.00607]（负向）。
- cv MF-SGD + Scale + Adam，C1 minus boundary_updated: -0.01326 [-0.01548, -0.01104]（负向）。
- cv MF-SGD + Adam，C1 minus boundary_updated: -0.00194 [-0.00328, -0.00060]（负向）。
- cv MF-SGD + AdaM³，C1 minus boundary_updated: -0.00930 [-0.01127, -0.00733]（负向）。
- cv MF-Momentum + AdaM³，C1 minus boundary_updated: -0.00751 [-0.00940, -0.00562]（负向）。
- cv MF-Momentum-Bias + AdaM³，C1 minus boundary_updated: -0.00659 [-0.00793, -0.00525]（负向）。
- nlp IID + Adam，C1 minus boundary_updated: -0.00092 [-0.00259, +0.00076]（区间包含零）。
- nlp MF-SGD + Scale + Adam，C1 minus boundary_updated: -0.00516 [-0.01173, +0.00141]（区间包含零）。
- nlp MF-SGD + Adam，C1 minus boundary_updated: +0.00138 [-0.00517, +0.00793]（区间包含零）。
- nlp MF-SGD + AdaM³，C1 minus boundary_updated: +0.00287 [-0.00143, +0.00716]（区间包含零）。
- nlp MF-Momentum + AdaM³，C1 minus boundary_updated: +0.00241 [-0.00209, +0.00691]（区间包含零）。
- nlp MF-Momentum-Bias + AdaM³，C1 minus boundary_updated: -0.00115 [-0.00577, +0.00348]（区间包含零）。
- cv method comparison，original: MF-SGD + Adam minus IID + Adam: +0.10034 [+0.09054, +0.11014]（正向）。
- cv method comparison，original: MF-SGD + Scale + Adam minus MF-SGD + Adam: +0.00387 [+0.00217, +0.00557]（正向）。
- cv method comparison，original: MF-SGD + AdaM³ minus MF-SGD + Adam: +0.02834 [+0.02528, +0.03140]（正向）。
- cv method comparison，original: MF-Momentum + AdaM³ minus MF-SGD + AdaM³: +0.03354 [+0.03182, +0.03526]（正向）。
- cv method comparison，original: MF-Momentum-Bias + AdaM³ minus MF-Momentum + AdaM³: +0.00671 [+0.00515, +0.00827]（正向）。
- nlp method comparison，original: MF-SGD + Adam minus IID + Adam: +0.01732 [+0.00979, +0.02484]（正向）。
- nlp method comparison，original: MF-SGD + Scale + Adam minus MF-SGD + Adam: +0.00929 [+0.00425, +0.01432]（正向）。
- nlp method comparison，original: MF-SGD + AdaM³ minus MF-SGD + Adam: +0.00000 [-0.00429, +0.00429]（区间包含零）。
- nlp method comparison，original: MF-Momentum + AdaM³ minus MF-SGD + AdaM³: +0.00252 [-0.00128, +0.00633]（区间包含零）。
- nlp method comparison，original: MF-Momentum-Bias + AdaM³ minus MF-Momentum + AdaM³: +0.00631 [+0.00316, +0.00946]（正向）。
- cv method comparison，boundary_updated: MF-SGD + Adam minus IID + Adam: +0.10034 [+0.09054, +0.11014]（正向）。
- cv method comparison，boundary_updated: MF-SGD + Scale + Adam minus MF-SGD + Adam: +0.00387 [+0.00217, +0.00557]（正向）。
- cv method comparison，boundary_updated: MF-SGD + AdaM³ minus MF-SGD + Adam: +0.02834 [+0.02528, +0.03140]（正向）。
- cv method comparison，boundary_updated: MF-Momentum + AdaM³ minus MF-SGD + AdaM³: +0.03354 [+0.03182, +0.03526]（正向）。
- cv method comparison，boundary_updated: MF-Momentum-Bias + AdaM³ minus MF-Momentum + AdaM³: +0.00671 [+0.00515, +0.00827]（正向）。
- nlp method comparison，boundary_updated: MF-SGD + Adam minus IID + Adam: +0.01732 [+0.00979, +0.02484]（正向）。
- nlp method comparison，boundary_updated: MF-SGD + Scale + Adam minus MF-SGD + Adam: +0.00929 [+0.00425, +0.01432]（正向）。
- nlp method comparison，boundary_updated: MF-SGD + AdaM³ minus MF-SGD + Adam: +0.00000 [-0.00429, +0.00429]（区间包含零）。
- nlp method comparison，boundary_updated: MF-Momentum + AdaM³ minus MF-SGD + AdaM³: +0.00298 [-0.00523, +0.01120]（区间包含零）。
- nlp method comparison，boundary_updated: MF-Momentum-Bias + AdaM³ minus MF-Momentum + AdaM³: +0.00573 [-0.00416, +0.01563]（区间包含零）。
- cv method comparison，C1: MF-SGD + Adam minus IID + Adam: +0.10601 [+0.09652, +0.11550]（正向）。
- cv method comparison，C1: MF-SGD + Scale + Adam minus MF-SGD + Adam: -0.00745 [-0.00894, -0.00596]（负向）。
- cv method comparison，C1: MF-SGD + AdaM³ minus MF-SGD + Adam: +0.02098 [+0.01686, +0.02510]（正向）。
- cv method comparison，C1: MF-Momentum + AdaM³ minus MF-SGD + AdaM³: +0.03533 [+0.03360, +0.03706]（正向）。
- cv method comparison，C1: MF-Momentum-Bias + AdaM³ minus MF-Momentum + AdaM³: +0.00763 [+0.00596, +0.00930]（正向）。
- nlp method comparison，C1: MF-SGD + Adam minus IID + Adam: +0.01961 [+0.01430, +0.02492]（正向）。
- nlp method comparison，C1: MF-SGD + Scale + Adam minus MF-SGD + Adam: +0.00275 [-0.00264, +0.00814]（区间包含零）。
- nlp method comparison，C1: MF-SGD + AdaM³ minus MF-SGD + Adam: +0.00149 [-0.00532, +0.00830]（区间包含零）。
- nlp method comparison，C1: MF-Momentum + AdaM³ minus MF-SGD + AdaM³: +0.00252 [-0.00780, +0.01285]（区间包含零）。
- nlp method comparison，C1: MF-Momentum-Bias + AdaM³ minus MF-Momentum + AdaM³: +0.00218 [-0.00850, +0.01286]（区间包含零）。

C=1 对照同时改变裁剪阈值及独立选择的学习率；算法对照同样使用各自最优学习率，因此属于完整训练配置比较，不能单独解释为某个机制的因果效应。MF、Scale、AdaM³ 的相对表现以上述配对差为依据。

诊断曲线见 figures/，裁剪比例来自每一步；更新范数、二阶矩和学习率记录前 20 步及之后每 10 步。历史未记录的优化器诊断明确标为缺失，不重跑补齐。所有新增训练保留检查点、训练曲线和噪声校准记录。

来源与审计：historical_sources_sha256.json、report_sources_sha256.json、source_provenance.json、scheduler.jsonl、failures.jsonl、final_audit.json。
