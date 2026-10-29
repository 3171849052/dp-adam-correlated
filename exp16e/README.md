# exp16e

仅 IID / B²LT，CV 与 NLP 各13方法，分别搜索，正式评价各10种子。所有新增文件只位于本目录，历史包从不导入。数据从 data/ 派生，本地分词器/模型从 cache/ 读取，禁用下载；重建输入张量与历史逐张量一致。

67次候选搜索：CV 9、旧NLP Adam 9、NLP STP-EX 8、Momentum 21、AdaM³ 20。新NLP八方法各取搜索种子前二，各加两个复核种子，共32次固定复核；三种子均值冻结后80次正式训练。旧方法仅当最佳新增候选在相同搜索种子上的准确率/损失排序不差于历史冻结配置时追加两个种子，共至多16次条件复核。合并历史全部有效三种子候选，以准确率均值降序、交叉熵均值升序、配置ID升序冻结。其余CV以及NLP IID直接保留历史冻结配置。变化的旧配置各重新做十种子，至多80次。核心179次、上限275次，严格同身份历史复用可减少。

STP-EX 是 exp16b 原公式：上一轮分母 d，scaled clip/noise 后 q；u=.9u+.1q；参数减 lr*u/(1-.9^t)；v=.999v+.001(d*q)^2，无末尾分母。Adam-STP 撤销缩放后运行标准Adam。Momentum原生未修正一阶矩，无Nesterov。AdaM³二阶矩输入是更新后未修正的一阶矩平方，与exp14a逐字一致。

协议：5 epochs、逻辑batch1000、FP32、全参数微调；CV物理250、搜索45000/5000 T225、正式50000/10000 T250；NLP物理1000、62000/5349搜索，正式相同62000及官方872，T310。五轮固定参与、GDP、epsilon8/delta1e-5、不使用抽样放大。搜索20261101、复核02/03、正式11–20。沿用历史初始化、固定顺序、增强与逐逻辑步dropout种子。矩阵只读exp16a已验收文件，按clip校准。

在仓库根目录运行（已有curve）：

```bash
CUDA_VISIBLE_DEVICES=1 /home/longt29/miniconda3/envs/curve/bin/python -B -m exp16e.stage1
bash exp16e/results/resume.sh
cat exp16e/results/background.json
cat exp16e/results/progress.json
cat exp16e/results/budget.json
tail -f exp16e/results/pipeline.log
```

start_background 创建独立session，以文件日志和继承flock脱离终端运行。GPU仅物理1/2/3，CV每卡最多一个worker，NLP每卡允许两个worker；worker内部cuda:0；NLP总并发最多6，其余卡可跑CV。发现外部占用时等待，不终止外部任务。

恢复入口为 results/resume.sh，保留锁阻止重复启动。恢复只复用完整身份/代码/输入/矩阵/完成清单及产物哈希一致的已封存trial。非有限搜索/复核记录原始traceback并排除，不重试；OOM、工程错误、未完成目录停止调度并保留现场，需人工诊断，不自动调整超参数、batch或精度。支持trial边界恢复，不承诺被中断trial的步级续训。

results/search_status.json、review_status.json、formal_status.json 保存每项状态、PID、GPU与日志路径；scheduler.jsonl 和 gpu_telemetry.jsonl 保存调度记录。完成后生成请求的九项报告文件及 theoretical_errors.csv。未完成时不生成虚假的最终统计。逐种子正式结果260项，历史复用来源逐项列出。配对区间是种子间Student-t95%区间，未作多重比较校正。算法各自调参，差异不是单一机制因果效应；单次epsilon8不代表搜索与重复运行联合epsilon8。

历史审阅：exp16a正式280次完成；exp16c CV改进后的Adam-SGD clip24优先于exp16a原clip17.3205；exp16c原生Momentum与STP-EX、exp16d AdaM³作为CV基准。exp16b NLP B²LT 27组粗搜索纳入STP-EX前二选取。exp14a只有单种子且MF机制为BandInvMF，因此不跨机制复用。exp16d固定参数专用复核不纳入本实验候选池，也不输出其结果。

用户后续授权NLP同卡两个进程。调度入口为 runtime/nlp_colocation.py，经同卡双进程真实模型smoke验收；仅替换调度capacity，不改动已封存训练源码。nlp_policy_transition.json记录旧worker自然结束、封存和调度器交接；nlp_colocation_acceptance.json记录同卡验收及调度脚本哈希。原start_background入口保留最初策略，当前启动和恢复统一使用results/resume.sh。

最新调度授权：GPU 1/2/3每卡最多两个NLP worker，总并发最多六个；CV每卡仍最多一个，CV与NLP不混占同一卡。当前入口为runtime/nlp_six.py，results/resume.sh调用此入口。nlp_six_acceptance.json记录六进程真实模型并发验收；nlp_six_transition.json记录自然结束和接续；最终审计记录该调度脚本哈希。此前nlp_colocation.py及其验收保留作为历史策略。

2026-10-29 正式输入准备故障恢复：67次搜索及34次复核已完成并冻结，协调器在正式输入准备阶段退出；内核日志记录libarrow/jemalloc后台线程段错误，原始证据保存在results/recovery_arrow/。正式872条输入用system Arrow内存池在独立进程重建并逐张量验证；协调器只核验已验证的输入文件，不再为缓存命中导入pandas/Arrow。最新恢复入口results/resume.sh调用runtime/supervised.py，保留nlp_six调度，新增父进程记录退出码/信号和faulthandler日志，不自动重试。background.json中pid为监督进程、pipeline_pid为协调器；background_exit.json仅在协调器结束后生成。
