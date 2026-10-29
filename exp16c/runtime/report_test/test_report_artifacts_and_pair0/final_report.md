# exp16c 最终报告

状态：正式训练及审计已完成。仅 CV / CIFAR-100 + ViT-Tiny。

各方法分别调参；下表是调参后性能比较，不是固定超参数下噪声机制的因果效应。历史测试集已被使用，不声称新盲测。

|方法|lr|clip|eps_scale|准确率均值 ± SD (%)|95% t CI|
|---|---:|---:|---:|---:|---|
|adam-mf-sgd-b2lt|0.5|3.0|None|60.450 ± 0.303|[60.233, 60.667]|
|adam-mf-momentum-bias-b2lt|0.5|3.0|None|61.450 ± 0.303|[61.233, 61.667]|
|adam-mf-momentum-b2lt|0.5|3.0|None|62.450 ± 0.303|[62.233, 62.667]|
|adam-mf-sgd-stp-b2lt|0.5|3.0|3.0|63.450 ± 0.303|[63.233, 63.667]|
|adam-iid|0.5|3.0|None|64.450 ± 0.303|[64.233, 64.667]|
|stpex-mf-momentum-bias-b2lt|0.5|3.0|3.0|65.450 ± 0.303|[65.233, 65.667]|
|momentum-iid|0.5|3.0|None|66.450 ± 0.303|[66.233, 66.667]|
|momentum-mf-sgd-b2lt|0.5|3.0|None|67.450 ± 0.303|[67.233, 67.667]|
|momentum-mf-momentum-b2lt|0.5|3.0|None|68.450 ± 0.303|[68.233, 68.667]|

## Momentum 三方法配对差（右减左，百分点）

- momentum-mf-sgd-b2lt − momentum-iid: 1.000, 95% CI [1.000, 1.000]
- momentum-mf-momentum-b2lt − momentum-iid: 2.000, 95% CI [2.000, 2.000]
- momentum-mf-momentum-b2lt − momentum-mf-sgd-b2lt: 1.000, 95% CI [1.000, 1.000]

选参证据见 `selection_evidence.json` 和 `top2.json`：搜索种子前二，追加 20261102/03，按三种子内部准确率均值、损失选优；两个 Adam 扩展保留原三种子最优作为基准。

`formal_per_seed.csv` 列出十种子的准确率、损失、五轮平均及末轮裁剪比例、耗时、显存和实际预算；`paired_differences.csv` 与 `paired_per_seed.csv` 给出所有方法同种子配对差及 t95% CI。区间未作多重比较校正。

`controlled_momentum.csv` 是共同 lr/clip/种子下的内部验证比较（搜索均有，复核仅取交集），不替代十种子正式受控实验。`momentum_comparison.csv` 是各自调参后的正式比较。

`matrix_errors.csv` 给出矩阵逆误差、分解误差、敏感度、目标函数和预算。`momentum_workload_errors.csv` 在共同实际 Momentum workload 下计算各机制噪声误差，已含 (1−β) 因子、clip²/1000²；参数误差另乘 lr²。这些线性噪声诊断不是训练误差。

单次 DP 训练 ε=8、δ=1e-5，固定五轮参与，GDP 校准，无采样放大；没有搜索、复核与重复运行的组合隐私声明。内部验证假设公开，裁剪诊断为内部审计产物。

完成数量：
```json
{
  "search": {
    "new_planned": 0,
    "new_completed": 0,
    "historical_completed": 0,
    "new_non_finite": 0
  },
  "review": {
    "new_planned": 0,
    "new_completed": 0,
    "historical_completed": 0,
    "new_non_finite": 0
  },
  "formal": {
    "new_planned": 0,
    "new_completed": 0,
    "historical_completed": 0,
    "new_non_finite": 0
  }
}
```
失败/数值排除记录：0，详见 final_audit.json。历史正式结果仅在配置完全相同时复用。所有历史来源只读校验通过。
