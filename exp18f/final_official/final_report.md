# Exp18f STP-Ex2 最终 NLP 正式实验

状态：已完成。新训练 10/10，复用 Exp18e 官方结果 20 条；更新时间 2026-11-06T17:22:24.556502+08:00。
两种 STP-Ex2 统一使用 lr=0.025、C=100、稳定系数 eps=0.001、β₁=0.9、β₂=0.999。
SST-2 / 本地 BERT-Tiny 全参数微调；FP32、关闭 TF32，长度128，批次1000，累积1，权重衰减0。
固定62000训练/5349内部验证划分，完整5轮×62步=310步。逐轮内部验证仅监控；无模型选择或提前停止。
872条官方验证仅评价第310步最终模型。所有模型和数据来自本地 cache/ 与 data/，全程离线。
种子顺序：20261111、20261112、20261113、20261114、20261115；六种方法逐种子初始化、训练排列和dropout种子核对一致。

## 六种算法官方验证总表

准确率、样本SD和置信区间均以百分数表示；SD使用ddof=1，95% CI使用五种子的Student-t(df=4)。

|算法|最优冻结参数 (lr, C, eps)|20261111|20261112|20261113|20261114|20261115|均值|样本SD|95% CI|平均验证损失|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Adam-IID|(0.002, 1, 1e-08)|76.261|76.261|74.771|77.294|76.147|76.147|0.899|[75.030, 77.263]|1.041504|
|Adam-B²LT-SGD|(0.01, 40, 1e-08)|79.931|78.211|77.523|79.358|77.408|78.486|1.120|[77.096, 79.876]|0.498466|
|Momentum-IID|(2, 1, 1e-08)|73.968|74.197|74.656|76.606|74.312|74.748|1.068|[73.422, 76.074]|1.091262|
|Momentum-B²LT-Momentum|(0.3, 20, 1e-08)|77.408|77.294|78.784|79.128|78.096|78.142|0.813|[77.132, 79.152]|0.532061|
|STP-Ex2-IID|(0.025, 100, 0.001)|73.968|77.523|75.115|78.326|74.656|75.917|1.898|[73.560, 78.274]|0.592135|
|STP-Ex2-B²LT-Momentum-Bias|(0.025, 100, 0.001)|78.784|79.128|77.638|79.817|78.440|78.761|0.808|[77.758, 79.765]|0.518551|

前四行完全复用 exp18e/formal/ 和 exp18e/per_seed.csv 的正式结果；后两行为本次新训练及官方评价。
Exp18f 搜索中的83.746%和79.787%是内部验证成绩，本表未将其用作官方成绩。

## 配对差异

以下均为后者减前者，单位为百分点；全部15组比较及损失差保存在 paired_statistics.json。
这些区间为描述性逐比较区间，未进行多重比较校正；Adam和Momentum的两种噪声方法超参数不同，不能仅归因于噪声机制。

|前者|后者|平均准确率差 (pp)|95% 配对CI (pp)|
|---|---|---:|---:|
|Adam-IID|Adam-B²LT-SGD|+2.339|[+1.207, +3.472]|
|Adam-IID|Momentum-IID|-1.399|[-2.575, -0.223]|
|Adam-IID|Momentum-B²LT-Momentum|+1.995|[+0.507, +3.484]|
|Adam-IID|STP-Ex2-IID|-0.229|[-2.192, +1.733]|
|Adam-IID|STP-Ex2-B²LT-Momentum-Bias|+2.615|[+2.306, +2.923]|
|Adam-B²LT-SGD|Momentum-IID|-3.739|[-5.401, -2.076]|
|Adam-B²LT-SGD|Momentum-B²LT-Momentum|-0.344|[-2.179, +1.491]|
|Adam-B²LT-SGD|STP-Ex2-IID|-2.569|[-5.164, +0.027]|
|Adam-B²LT-SGD|STP-Ex2-B²LT-Momentum-Bias|+0.275|[-0.812, +1.362]|
|Momentum-IID|Momentum-B²LT-Momentum|+3.394|[+2.624, +4.165]|
|Momentum-IID|STP-Ex2-IID|+1.170|[-0.531, +2.871]|
|Momentum-IID|STP-Ex2-B²LT-Momentum-Bias|+4.014|[+2.902, +5.126]|
|Momentum-B²LT-Momentum|STP-Ex2-IID|-2.225|[-4.471, +0.021]|
|Momentum-B²LT-Momentum|STP-Ex2-B²LT-Momentum-Bias|+0.619|[-0.803, +2.042]|
|STP-Ex2-IID|STP-Ex2-B²LT-Momentum-Bias|+2.844|[+1.062, +4.627]|

## 运行与审计

后台调度PID：3521571；总日志：logs/pipeline.log。正式工作进程PID见 logs/formal_events.jsonl。
冒烟测试：三卡各运行两方法3步（另计6次），检查完整日志、有限参数、配对初始化/排列和显存；正式训练仅10次。
- GPU 0: NVIDIA GeForce RTX 3080 Ti，`GPU-bb24c6dd-b3e9-d35e-f544-bb53c11078c6`。
- GPU 1: NVIDIA GeForce RTX 3080 Ti，`GPU-015dd2bc-e3fc-8a3a-9ec6-ae5664031a54`。
- GPU 2: NVIDIA GeForce RTX 3080 Ti，`GPU-9cee712b-b39b-6d0a-29fb-1aaeb8bffa8e`。

正式训练最大单进程峰值预留显存：4092 MiB；每卡最多2进程，UUID隔离，无DDP。
每次训练的 config.json、pairing.json、privacy.json、steps.jsonl、curve.json、worker.log、checkpoint.pt、summary.json 和 complete.json 位于 formal/<method>_seed<seed>/。
protocol.json、hyperparameters.json、data_audit.json、matrix_audit.json、privacy_audit.json、smoke_verification.json 和 integrity_audit.json 保存协议和审计证据。
input_manifest.json 与 code_manifest.json 固定原输入及本次执行代码；最终审计确认原Exp18e与Exp18f搜索/冻结输入未被修改。

每次训练独立校准隐私 ε=8、δ=1e-5；增删固定零槽邻接，参与5次、间隔62步，无抽样放大。
沿用Exp18e的B²LT矩阵(T=310,K=P=62,Q=5)；C和D的逆关系、工作负载重构及敏感度独立核验。
IID噪声和标准差134.21530075084846，B²LT为229.321310727543；对应批平均标准差为0.13421530075084845和0.229321310727543。
稳定系数eps与隐私epsilon不同；多次训练、搜索和正式实验的组合不具备联合ε=8保证；原始内部监控量不是DP发布。

复现：使用 Conda curve 环境，在仓库根目录运行 `python -B -m exp18f.final_official.start_background`（新输出目录首次启动），或运行 `python -B -m exp18f.final_official.report` 重新核验汇总。
