# Exp7b

七种 DP-Adam 方法，复用 Exp2 的模型、fixed-epoch GDP 校准、BandInvMF FIR noise 和 Scale 实现。所有新增文件在 `exp7b/`，Exp1–Exp7 只读。使用 conda `curve` 和物理 GPU **1、2、3**；GPU 0 留给现有任务。

固定协议：CIFAR-100，仓库 `data/`；ViT-Tiny pretrained 参数显式读取仓库 `cache/`，offline、禁止下载；full fine-tuning；5 epochs、50000 examples、logical batch 1000、physical batch 250、250 optimizer/noise steps；Adam β=(.9,.999)、eps=1e-8、weight_decay=0；ε=8、δ=1e-5、add/remove zero-out、无采样放大、fixed participation k=5 / spacing=50；4-band BandInvMF。

## 入口

从仓库根目录运行 Stage 1（unit tests + 七种算法各 1 个 GPU smoke）：

```bash
conda run --no-capture-output -n curve python -B -m exp7b.stage1 --gpus 1 2 3
```

Smoke 使用完整模型、4 个 250-example microbatches、1 个 logical step、100 个测试样本，校准仍按 250-step 协议。Smoke 不能作为 epoch-5 搜索证据，不代表 full-run accuracy 或 full-run 数值稳定性。

Stage 2 单条入口（自动有界搜索 → 冻结 → 70 个 full runs → 报告）：

```bash
conda run --no-capture-output -n curve python -B -m exp7b.stage2 --gpus 1 2 3
```

Stage 1 不会启动 Stage 2。Stage 2 重新检查 tests/smoke，复用已完成 smoke。日志是 `results/stage2.log`。同时只运行一个 launcher；每张 GPU 最多一个 trial。全局 FIFO 依照候选列表启动，不按算法分配独立 GPU 队列；跨队列轮换首选 GPU，final 按 seed 轮换算法顺序。无恢复框架或 GPU 锁；完整结果按 canonical trial ID 复用，numerical failure 记录后排除，其他失败/不完整目录直接报错供检查。

独立入口：`exp7b.search` 仅搜索与冻结；`exp7b.final` 仅读取 frozen settings 并跑 final；`exp7b.report --final` 重建最终报告；`exp7b.audit` 检查已有 artifacts。都用 `python -B -m` 调用。

单 trial 入口例如：

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n curve python -B -m exp7b.train --method dp-adam-momentum-bias-bandinvmf --seed 20261001 --lr .005 --max-grad-norm 30 --smoke
```

## Workload 与 geometry

| Method | Workload | Query geometry |
|---|---|---|
| dp-adam-iid | identity strategy / iid noising | standard |
| dp-adam-sgd-bandinvmf | prefix/SGD | standard |
| dp-adam-momentum-bandinvmf | Exp2 momentum | standard |
| dp-adam-momentum-bias-bandinvmf | bias-corrected first-moment cumulative workload | standard |
| dp-adam-sgd-bandinvmf-scale | prefix/SGD | Scale |
| dp-adam-momentum-bandinvmf-scale | Exp2 momentum | Scale |
| dp-adam-momentum-bias-bandinvmf-scale | bias-corrected first-moment cumulative workload | Scale |

Momentum-Bias（1-indexed）:

`W[t,j] = sum(beta1**(k-j)/(1-beta1**k), k=j..t)`，忽略全局常数 `(1-beta1)`。

该 W 为完整 **250×250 non-Toeplitz** 矩阵。令 `D` 为 4-band lower-triangular Toeplitz noising filter，strategy `S=D^{-1}`。新 filter 以两个固定初值进行 CPU float64 Nelder-Mead，固定 d0=1（全局 filter scale 在目标里抵消），最小化 `fixed_epoch_sensitivity(S,5,50)^2 * ||W D||_F^2 / 250`。用所有 W entries 构造二次型来加速评价；最后直接用完整矩阵和 Exp2 accountant 复核目标。隐私仍由实际 S 的 fixed-participation absolute Gram bound 校准。SGD/Momentum 保持 Exp2 coefficients，不重新优化；同 workload 的两种 geometry 共享 filter。

Scale 用 `s_t=1/(sqrt(vhat_(t-1))+eps_scale)`，冻结上一步 second moment，按 `||s_t*g_i||` clipping，在 scaled sum space 添加 FIR Gaussian noise，再 inverse-scale，除以 1000 后送入普通 Adam。Adam eps=1e-8 与 eps_scale 分开。Bias 指一阶动量 bias correction workload，不是 Algorithm 7 的二阶矩修正算法。

## 有界搜索与冻结

Search seed 固定 20261001，唯一 selection objective 为 epoch-5 `final_test_top1`。同分以 canonical trial ID 稳定排序。诊断不参与打分。

Exp7 冻结配置从 `exp7/results/selected_configs.json` 导入 `results/exp7_frozen_configs.json` 并断言：IID (.0005,30)、Momentum (.005,30)、SGD-Scale (.002,100,.1)。它们不会重新搜索。

四个搜索方法：

- SGD：C=10，LR=.001/.002/.003；内部 winner 停止，边界只扩一次。
- Momentum-Scale：LR=.005，eps=.03/.1/.3/1、K=C*eps=10；审计复用现有 C=50/100/200 与 LR=.003/.005/.007 证据。eps=.1 保持最好且 bracket 成立则冻结，否则在 eps 邻域搜索 K=5/10/20，再检查 LR=.003/.005/.007。eps/LR 边界至多扩一次。
- Momentum-Bias：LR=.005，C=3/10/30/100；C 边界至多补 1 或 300；最佳 C 下 LR=.002/.003/.005/.007/.01，边界至多扩一次。
- Momentum-Bias-Scale：LR=.005、K=10，eps=.03/.1/.3/1；最佳 eps 与邻域检查 K=5/10/20；eps 和 K 边界各至多一次 refinement，然后最佳 (eps,C) 下检查 LR=.002/.003/.005/.007/.01、至多一次 LR 扩展。

K 只用于生成 C，不传给训练器。搜索每轮保存 candidate JSON、summary CSV/JSON 和 report。候选优先审计复用兼容 Exp2/Exp7 实际 artifacts（协议、初始化、checkpoint、permutation、coefficients、strategy、full workload、noise 校准与有限 checkpoint）。原结果保持只读，在 exp7b 保存规范化副本。无兼容证据才训练，不把提示中的数值当成已完成结果。

四个搜索方法各按其所有实际完成 search trials 的 Top-1 选 winner，写 `results/search/selected_configs.json`；合并七方法到 `results/frozen_configs.json`，保存 `frozen_manifest.json` SHA256。已生成的 frozen settings 不可覆盖为新设置。Final runner 和单-trial final 入口校验 manifest、精确匹配 lr/C/eps_scale；final 不搜索。

## 配对、审计和输出

Final seeds 20261011–20261020；每个 seed 全部七种方法共享模型/classifier 初始化、一次随机 permutation（各 epoch 重用，保证 spacing=50）和 DataLoader worker augmentation RNG 约定。Noise 使用独立 seed+1 generator；增强由独立 seed generator 提供。保存训练顺序与每 logical batch 前四个 transformed examples 的 rolling SHA256，逐 epoch 比较同 seed 所有方法。

- `results/smoke/trials/<id>/`：smoke artifacts。
- `results/search/trials/<id>/`：search artifacts / 历史规范化副本。
- `results/final/trials/<id>/`：仅 frozen configs 的 final artifacts。
- 每 trial：`config.yaml`、`train.log`、`metrics.csv`、`mechanism_metrics.csv`、`matrices.npz`、`train_order.npy`、`final.pt`、`summary.json`。审计 step/epoch 数、ε/δ、finite model/optimizer、workload/strategy/matrix/checkpoint hash、pretrained hash、seed/permutation 和 pairing。
- Final：`results/final_multiseed.csv`、`method_summary.json`、`paired_effects.csv/json`、`final_report.md`；raw Top-1、mean、sample std(ddof=1)、SE、t(df=9) 95% CI；八项 paired comparisons 与 wins/10；三 workload × 两 geometry 结构化比较。

报告中 “best method − IID” 依据 final mean 做描述性比较，未进行多重比较校正；它不用于调超参数。Stage 1 的通过证据在 `results/unit_tests.log`、`platform_validation.json`、`bias_matrix_audit.json` 和 smoke artifacts。
