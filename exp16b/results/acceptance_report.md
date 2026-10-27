# 验收结果

29 项测试通过；8 次真实模型 smoke 全部成功；4 组重复训练模型张量逐个完全一致。

四套矩阵通过已验收来源哈希、Momentum-Bias 定义、逆关系、敏感度、GDP、完整时域 CPU/GPU 噪声流及经验协方差检查。

逐样本裁剪与梯度累积对照独立 autograd；包含重复 token 及 padding。原版 STP-EX 公式逐步对照；非有限更新拒绝。

GPU 0 独占 CV 或两进程 NLP；全参数 FP32，CV物理250、NLP物理1000。

首次验收因修正配置元数据触发代码哈希变化，现场完整保留于 runtime/acceptance_before_metadata_fix；最终代码重新通过全部验收。无协议更改。

|任务|机制|重复|峰值分配MiB|两步用时秒|
|---|---|---:|---:|---:|
|cv|stpex-mf-momentum-bias-bandinvmf|0|8178.1|4.83|
|cv|stpex-mf-momentum-bias-b2lt|0|8262.8|4.99|
|nlp|stpex-mf-momentum-bias-bandinvmf|0|3839.0|0.79|
|nlp|stpex-mf-momentum-bias-b2lt|0|3906.3|0.84|
|cv|stpex-mf-momentum-bias-bandinvmf|1|8178.1|4.77|
|cv|stpex-mf-momentum-bias-b2lt|1|8262.8|4.95|
|nlp|stpex-mf-momentum-bias-bandinvmf|1|3839.0|0.82|
|nlp|stpex-mf-momentum-bias-b2lt|1|3906.3|0.76|
