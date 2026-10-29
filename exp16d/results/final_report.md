# exp16d 最终报告

状态：40 次正式训练完成。CIFAR-100 / ViT-Tiny，FP32，5 epochs。

|方法|lr|clip|准确率均值 ± SD (%)|95% t CI|loss|裁剪比例|秒|峰值显存 GiB|
|---|---:|---:|---:|---|---:|---:|---:|---:|
|adam3-iid|0.00015|30|59.194 ± 1.487|[58.130, 60.258]|1.4965|0.9585|692.8|8.13|
|adam3-mf-sgd-b2lt|0.0004|20|77.207 ± 0.464|[76.875, 77.539]|0.7665|0.8048|707.0|8.26|
|adam3-mf-momentum-b2lt|0.0007|10|75.348 ± 0.667|[74.871, 75.825]|0.8341|0.8783|699.4|8.26|
|adam3-mf-momentum-bias-b2lt|0.0004|20|77.240 ± 0.455|[76.914, 77.566]|0.7680|0.8019|706.4|8.26|

## B²LT workload 正式比较（各自调参）
- adam3-mf-momentum-b2lt − adam3-mf-sgd-b2lt: -1.859 pp，95% CI [-2.073, -1.645]。
- adam3-mf-momentum-bias-b2lt − adam3-mf-sgd-b2lt: 0.033 pp，95% CI [-0.021, 0.087]。
- adam3-mf-momentum-bias-b2lt − adam3-mf-momentum-b2lt: 1.892 pp，95% CI [1.680, 2.104]。

## 公共配置内部验证（三种子，lr=0.0004 / clip=20）
- adam3-mf-momentum-b2lt − adam3-mf-sgd-b2lt: -3.200 pp，95% CI [-4.973, -1.427]。
- adam3-mf-momentum-bias-b2lt − adam3-mf-sgd-b2lt: -0.007 pp，95% CI [-0.337, 0.324]。
- adam3-mf-momentum-bias-b2lt − adam3-mf-momentum-b2lt: 3.193 pp，95% CI [1.719, 4.667]。
三种子公共配置不作为十种子正式结论。

## 与 exp16a 标准 Adam 对照
- adam3-mf-sgd-b2lt: AdaM³ 77.207% vs Adam 76.995%，同种子差均值 0.212 pp。Adam lr=0.003, clip=17.3205080757。
- adam3-mf-momentum-b2lt: AdaM³ 75.348% vs Adam 70.860%，同种子差均值 4.488 pp。Adam lr=0.007, clip=10.0。
- adam3-mf-momentum-bias-b2lt: AdaM³ 77.240% vs Adam 76.921%，同种子差均值 0.319 pp。Adam lr=0.003, clip=17.3205080757。

不同超参数的比较不能直接解释为优化器或 workload 本身的因果效应。历史官方测试集已使用，不声称新盲测。区间是种子间 Student-t 区间，未作多重比较校正。

单次训练 ε=8、δ=1e-5，固定五轮参与，无采样放大，GDP 校准；每个 clip 独立校准。全部搜索与重复训练不具有联合 ε=8 保证。内部验证及训练诊断按内部研究数据处理。

隐私数值见 privacy_calibration.csv；矩阵校验见 matrix_audit.json；选参证据见 selection_evidence.json；逐 trial 日志、checkpoint、耗时及显存保存在对应目录。
