# 与 Exp9 / Exp14 的差异

|项目|历史真实实现|Exp15|
|---|---|---|
|优化器|Exp9 为 Adam；Exp14 的 AdaM3 仍除以当前二阶矩分母|Momentum 系列直接更新；STP-EX 系列在预条件化空间做带偏差修正动量，直接使用 uhat 更新|
|Scale|上一时刻 vhat 定义裁剪尺度，加噪后撤销缩放再交给 Adam|裁剪与加噪后保留 qtilde，仅 stpex 二阶矩输入撤销缩放|
|二阶矩|Adam 使用带噪梯度平方；AdaM3 使用未校正 m 平方|stpex: (d_previous*qtilde)^2；stpex2: qtilde^2；stpex-adam3: 未校正 u^2|
|分母|历史 Scale eps_scale 随配置；Adam 自身 eps=1e-8|STP-EX d=sqrt(vhat)+eps_s，eps_s 固定读取 Exp9 Momentum-Bias Scale 的 CV=0.05/NLP=0.01；v0=0，故 d0=eps_s；无额外 Adam 分母|
|噪声工作负载|Exp9 对不同工作负载分别设计滤波器|主对照全部使用已保存的 Momentum-Bias 四频带系数；单独记录 momentum 的真实工作负载，不重新优化滤波器|
|裁剪阈值|历史使用 C，Scale 在缩放空间定义|统一配置名 clip；Momentum 解释为 C，STP-EX 解释为 R|
|搜索|历史网格与方法不同|先摸底再冻结；A 固定阈值 1；B 联合三阈值和三学习率；两方案复用重合训练|
|数据与模型|CIFAR100 + ViT-Tiny；SST2 + BERT-tiny|保持相同离线数据、完整可训练 FP32 参数、预处理与模型权重映射|
|步数与批次|CV 搜索 225 步/正式 250 步；NLP 均 310 步；逻辑批次 1000，物理批次 CV=250/NLP=1000|完全沿用；验收仅执行前两步，摸底前 20 步；完整目标矩阵校准后取前缀隐私预算|
|评价|内部验证用于搜索，官方评价用于最终复跑|沿用；验收只评价 64 个内部验证样本；无官方选参|
|随机流|初始化/数据顺序同 seed；噪声独立 seed+1；NLP dropout 每步固定|保持相同参数遍历、初始化、顺序、CV DataLoader 流和 NLP dropout 流；正式种子从 Exp9 代码读取|
|诊断|历史部分只记录前 20 步及每 10 步|每逻辑步记录损失、裁剪比例、裁剪空间梯度范数、带噪梯度范数、更新范数、分母和二阶矩统计、噪声与 epsilon|

独立复制了 Exp9 的裁剪/噪声/会计代码及 Exp14a 的模型/数据代码，并改为 Exp15 本地导入。复制前的 SHA256 见 `results/copied_sources.json`，每次训练记录当前实际源文件哈希和 Git HEAD。复制并不意味着原样使用历史 Scale/Adam 的更新：Exp15 已显式替换其逻辑批次与优化器接口。
