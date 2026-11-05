# Exp18b 输入验收：阻塞，未启动

检查时间：2026-11-05T13:42:42.850403+08:00。环境：`/home/longt29/miniconda3/envs/curve/bin/python`。

MNIST 数据检查通过，train.pt、test.pt、split.npz 的 SHA256 与 Exp18 完全一致；划分为 54000/6000/10000。未下载数据，未进行测试集评价。实际绝对路径、SHA256、形状和类型见 data_audit.json。

cache 中只有 BERT-tiny 和 ViT 两份模型权重（ViT snapshot 是 blob 的符号链接）。BERT 有 49 个张量，ViT 有 152 个张量，均缺少 Exp18 CNN 的全部六个参数键。ViT 的 patch_embed.proj.weight 为 [192,3,16,16]，分类头为 [1000,192]；Exp18 首层为 [8,1,5,5]，分类头为 [10,784]。全部实际键名、维度、路径和 SHA256 见 checkpoint_audit.json；cache 全文件清单见 cache_inventory.json。

按用户要求在权重不兼容时停止。未实现或启动后续训练流水线，未运行六方法 smoke test、隐私校准、矩阵验收或三卡并行验收，未冻结训练协议。实际加载的 CNN checkpoint：无。后台 PID：无。GPU 分配：无。完成训练：0/90。未生成虚构的指标、检查点、GPU 遥测或通过记录。

Exp18 使用随机初始化；本任务要求统一使用兼容 cache 权重，因此不能回退到 Exp18 的随机初始化，也不能替换为 ViT/BERT。恢复需要在 cache/ 提供与 Exp18 CNN 完全兼容的真实 checkpoint，再复查输入并继续实现和验收。现有脚本仅进行离线输入审计，不启动训练。

复查命令：`/home/longt29/miniconda3/envs/curve/bin/python -B -m exp18b.preflight`（缺少兼容权重时退出码为 2）。执行日志：preflight.log。

隐私协议说明：要求每次完整训练 ε=3、δ=1e-5、GDP 无采样放大；90 次训练的组合隐私预算不能声明为 ε=3。当前没有执行训练，也未声称完成噪声标准差校准。
