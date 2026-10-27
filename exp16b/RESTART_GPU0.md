# GPU 0 实验停止与恢复

**2026-10-27 更新：当前已改为后台排在 exp17 后使用 GPU 2、3，见 [最新恢复说明](RESTART_GPUS_2_3.md)。以下为此前停止 GPU 0 时的历史记录。**

2026-10-26 18:42（Asia/Shanghai）按用户要求停止 exp16b 调度器 PID 862590、训练进程 PID 2602899 及其数据加载子进程。GPU 0 已释放，GPU 1–3 的 exp17 训练继续运行。尚未重新启动 exp16b。

108 个粗搜索试验中，37 个完成且通过恢复入口的配置、代码、输入资产、矩阵及产物哈希校验；余下 71 个待运行，包含此次中断的试验。

中断试验 `359aaae1240908c02cb6`（CV、stpex-mf-momentum-bias-b2lt、lr=1.5、clip=1、eps_scale=0.1、seed=20261101）最后记录到第 28/225 步，没有中途 checkpoint。恢复时从第 0 步重跑该试验，复用已有 37 个完整结果。原始现场已整体移动至归档目录，避免原调度器因未完成目录而拒绝恢复；没有删除现场或修改已完成试验。

在仓库根目录运行：

```bash
# 仅校验，不启动；本次已成功执行。
bash exp16b/resume_gpu0.sh --check

# 以后需要恢复时运行；完成校验后启动后台单实例调度器。
bash exp16b/resume_gpu0.sh

# 查看运行信息和进度。
cat exp16b/results/background.json
cat exp16b/results/progress.json
tail -f exp16b/results/pipeline.log
```

脚本固定使用原 curve Python 环境及 CUDA_VISIBLE_DEVICES=0。既有单实例锁继续生效；GPU 0 有外部计算进程时，调度器会等待。

停止清单、原启动参数和必要环境、停止前状态及中断试验文件位于：

`exp16b/runtime/stopped_gpu0_20261026_184212/`

`exp16b/runtime/latest_gpu0_stop.txt` 保存此次归档的绝对路径。当前进度状态记录为 37 completed、1 interrupted、70 pending；恢复后调度器会将中断试验重新列为 pending。
