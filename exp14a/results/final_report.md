# Exp14a: MF + AdaM³ 对照 Exp9

固定种子 20261111；ε=8，δ=1e-5；冻结学习率和裁剪阈值。仅将二阶矩输入改为未偏差修正的一阶矩平方。

|任务|机制|状态|Exp9 准确率|AdaM³ 准确率|差值（百分点）|
|---|---|---|---:|---:|---:|
|cv|dp-adam-iid|completed|59.8600%|21.8200%|-38.040|
|cv|dp-adam-sgd-bandinvmf|completed|70.1500%|12.7700%|-57.380|
|cv|dp-adam-momentum-bandinvmf|completed|73.0900%|13.9700%|-59.120|
|cv|dp-adam-momentum-bias-bandinvmf|completed|76.9400%|17.1200%|-59.820|
|nlp|dp-adam-iid|completed|76.3761%|66.2844%|-10.092|
|nlp|dp-adam-sgd-bandinvmf|completed|76.8349%|50.9174%|-25.917|
|nlp|dp-adam-momentum-bandinvmf|completed|75.3440%|49.0826%|-26.261|
|nlp|dp-adam-momentum-bias-bandinvmf|completed|76.6055%|49.0826%|-27.523|

单种子结果仅用于描述性配对比较，不做统计显著性推断。历史评估集并非新的盲测集。

所有已完成训练均核对初始化、数据顺序、增强/Dropout 随机轨迹和矩阵哈希与 Exp9 相同。训练耗时包括诊断开销。

详见 summary.csv、comparison.csv、training_curves.png 和各任务目录。
