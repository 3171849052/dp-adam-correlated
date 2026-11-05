import json

import numpy as np
from scipy.stats import t

from exp18c import BASE, ROOT
from exp18c.common import (GROUPS, GRID, SEEDS, completed, now, read, save, sha,
                           trial_dir, trial_id, write_csv)


from exp18b.report import stats


def verify_pairing(rows):
    for group, methods in GROUPS.items():
        for lr in sorted({r["config"]["lr"] for r in rows if r["config"]["group"] == group}):
            for seed in sorted({r["config"]["seed"] for r in rows if r["config"]["group"] == group}):
                pair = [r for r in rows if r["config"]["group"] == group
                        and r["config"]["lr"] == lr and r["config"]["seed"] == seed]
                assert len(pair) == 2 and {r["config"]["method"] for r in pair} == set(methods)
                assert pair[0]["pairing"] == pair[1]["pairing"], "Initial/order/innovation pairing differs"
                assert pair[0]["final_innovation_rng_sha256"] == pair[1]["final_innovation_rng_sha256"]


def select(rows):
    assert len(rows) == 90
    rows = sorted(rows, key=lambda row: trial_id(row["config"]))
    verify_pairing(rows)
    flat = []
    for row in rows:
        cfg = row["config"]
        flat.append(dict(trial_id=trial_id(cfg), **cfg, validation_accuracy=row["metrics"]["accuracy"],
            validation_loss=row["metrics"]["loss"], final_train_loss=row["train_loss"],
            train_seconds=row["train_seconds"], wall_seconds=row["seconds"],
            peak_memory_mib=row["peak_memory"] / 2 ** 20,
            examples_per_second=row["training_examples_per_second"]))
    write_csv(BASE / "grid_per_seed.csv", sorted(flat, key=lambda r: r["trial_id"]))
    aggregate = []
    best = {}
    for group, methods in GROUPS.items():
        ranking = []
        for lr in GRID[group]:
            pair_rows = [r for r in flat if r["group"] == group and r["lr"] == lr]
            assert len(pair_rows) == 10
            ranking.append(dict(lr=lr, pair_validation_accuracy=float(np.mean([r["validation_accuracy"] for r in pair_rows])),
                                pair_validation_loss=float(np.mean([r["validation_loss"] for r in pair_rows]))))
            for method in methods:
                method_rows = [r for r in pair_rows if r["method"] == method]
                aggregate.append(dict(group=group, method=method, lr=lr,
                    **{f"{metric}_{key}": value for metric in ("validation_accuracy", "validation_loss", "train_seconds")
                       for key, value in stats([r[metric] for r in method_rows]).items()}))
        ranking.sort(key=lambda r: (-r["pair_validation_accuracy"], r["pair_validation_loss"], r["lr"]))
        best[group] = dict(lr=ranking[0]["lr"], clip=pair_rows[0]["clip"],
                          eps=pair_rows[0]["eps"], methods=methods, ranking=ranking)
    write_csv(BASE / "grid_summary.csv", aggregate)
    selection = dict(frozen=True, rule=read(BASE / "protocol.json")["selection"],
        groups=best, seeds=SEEDS, grid_trials=90, official_test_used_for_selection=False,
        selection_inputs={trial_id(r): sha(trial_dir(r) / "summary.json") for r in [row["config"] for row in rows]},
        protocol_sha256=sha(BASE / "protocol.json"))
    # The timestamp is separate so resumption can compare exact selection content.
    path = BASE / "best_hyperparameters.json"
    if path.exists():
        assert read(path) == selection, "Frozen hyperparameters changed"
    else:
        save(path, selection)
        save(BASE / "selection_freeze.json", dict(time=now(), sha256=sha(path)))
    return [row for row in rows if row["config"]["lr"] == best[row["config"]["group"]]["lr"]]


def report(selected, evaluations):
    assert len(selected) == len(evaluations) == 30
    per_seed = []
    by_id = {r["trial_id"]: r for r in evaluations}
    for row in selected:
        cfg = row["config"]
        ev = by_id[trial_id(cfg)]
        assert ev["checkpoint_sha256"] == sha(trial_dir(cfg) / "checkpoint.pt")
        records = [json.loads(line) for line in (trial_dir(cfg) / "steps.jsonl").read_text().splitlines()]
        per_seed.append(dict(trial_id=trial_id(cfg), group=cfg["group"], method=cfg["method"],
            seed=cfg["seed"], lr=cfg["lr"], clip=cfg["clip"], eps=cfg["eps"],
            validation_accuracy=row["metrics"]["accuracy"], validation_loss=row["metrics"]["loss"],
            test_accuracy=ev["metrics"]["accuracy"], test_loss=ev["metrics"]["loss"],
            final_train_loss=row["train_loss"], mean_train_loss=float(np.mean([r["loss"] for r in records])),
            train_seconds=row["train_seconds"], wall_seconds=row["seconds"],
            validation_seconds=row["validation_seconds"], test_seconds=ev["seconds"],
            peak_memory_mib=row["peak_memory"] / 2 ** 20,
            peak_reserved_mib=row["peak_reserved"] / 2 ** 20,
            examples_per_second=row["training_examples_per_second"], physical_gpu=row["physical_gpu"],
            checkpoint_sha256=ev["checkpoint_sha256"]))
    per_seed.sort(key=lambda r: (r["method"], r["seed"]))
    write_csv(BASE / "formal_per_seed.csv", per_seed)
    metrics = ["test_accuracy", "test_loss", "validation_accuracy", "validation_loss", "final_train_loss",
               "mean_train_loss", "train_seconds", "wall_seconds", "peak_memory_mib", "examples_per_second"]
    summaries, paired = [], []
    for group, methods in GROUPS.items():
        by_method = {method: {r["seed"]: r for r in per_seed if r["method"] == method} for method in methods}
        for method in methods:
            assert set(by_method[method]) == set(SEEDS)
            row = dict(group=group, method=method, lr=by_method[method][SEEDS[0]]["lr"])
            row.update({f"{metric}_{key}": value for metric in metrics
                        for key, value in stats([by_method[method][s][metric] for s in SEEDS]).items()})
            summaries.append(row)
        for metric in metrics:
            diffs = [by_method[methods[1]][s][metric] - by_method[methods[0]][s][metric] for s in SEEDS]
            paired.append(dict(group=group, metric=metric, direction="B2LT minus IID",
                               **{f"seed_{s}": value for s, value in zip(SEEDS, diffs)}, **stats(diffs)))
    write_csv(BASE / "summary.csv", summaries)
    write_csv(BASE / "paired_differences.csv", paired)
    lines = ["# Exp18c 完整实验报告", "", f"完成时间：{now()}。90/90 完整训练，30/30 入选检查点官方测试评价。", "",
        "模型与 Exp18 相同：两层卷积小型 CNN，11,274 个参数。逐字节复用 Exp18b 的五份初始化；"
        "未使用 cache 中不兼容的 BERT/ViT 权重。五份按种子生成的初始化保存在 initializations/，"
        "每份由同一种子的六种方法和三个学习率共用。它们是随机初始化文件，不是预训练 cache 权重。", "",
        "MNIST 54000/6000/10000，3 epoch、162 步，逻辑/物理 batch 1000、累积 1、FP32、无权重衰减。"
        "三个学习率完整网格，各优化器的 IID/B2LT 使用相同超参数。验证集最终准确率在两种噪声与五种子间等权平均选择；"
        "冻结 best_hyperparameters.json 后，每个入选种子检查点进行一次完整官方测试评价，无额外训练、无测试集调参。", "",
        "隐私：每次完整训练 ε=3、δ=1e-5，GDP、增删零槽邻接、无采样放大。三份 B2LT 使用 Exp18 的原矩阵，"
        "T=162、K=P=54、Q=5；按本次 clip 重新计算噪声。90 次训练的组合隐私预算不等于 ε=3。"
        "验证集视为公开；原始训练损失与裁剪诊断是内部审计信息，不声明为额外 DP 发布。", "",
        "以下为五种子均值、样本标准差与 Student-t 95% CI（df=4），未作多重比较校正。", "",
        "| 方法 | LR | 测试准确率均值 | SD | 95% CI | 测试损失均值 |", "|---|---:|---:|---:|---|---:|"]
    for r in summaries:
        lines.append(f"| {r['method']} | {r['lr']} | {r['test_accuracy_mean']:.6f} | {r['test_accuracy_sd']:.6f} | "
                     f"[{r['test_accuracy_ci95_low']:.6f}, {r['test_accuracy_ci95_high']:.6f}] | {r['test_loss_mean']:.6f} |")
    lines += ["", "IID/B2LT 配对差异（B2LT − IID，测试准确率）：", ""]
    for r in paired:
        if r["metric"] == "test_accuracy":
            lines.append(f"- {r['group']}：均值 {r['mean']:.6f}，SD {r['sd']:.6f}，95% CI [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}]。")
    lines += ["", "| 方法 | 训练秒数均值 ± SD | 训练时间 95% CI | 样本/秒均值 | CUDA 峰值 MiB 均值 | 平均训练损失 |",
              "|---|---:|---|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['method']} | {r['train_seconds_mean']:.3f} ± {r['train_seconds_sd']:.3f} | "
            f"[{r['train_seconds_ci95_low']:.3f}, {r['train_seconds_ci95_high']:.3f}] | {r['examples_per_second_mean']:.1f} | "
            f"{r['peak_memory_mib_mean']:.1f} | {r['mean_train_loss_mean']:.6f} |")
    lines += ["", "训练计时排除验证与进程启动；wall_seconds 包含初始化、数据加载、验证与检查点保存。"
        "并发下每个 trial 的速度受同时运行进程影响，速度配对差异不能直接解释为单进程算法开销。"
        "gpu_telemetry.csv 记录自动检测的四张物理 GPU 的利用率、显存和调度容量。", "",
        "所有损失、验证指标、效率与显存统计的完整均值、SD 和 CI 见 summary.csv；"
        "逐种子指标见 formal_per_seed.csv；所有配对指标及逐种子差值见 paired_differences.csv；"
        "完整网格见 grid_per_seed.csv/grid_summary.csv。配置、曲线、隐私、配对哈希和检查点在 trials/。", "",
        "冻结的代码/配置哈希见 acceptance.json，数据路径与 SHA256 见 data_audit.json，"
        "随机初始化路径与 SHA256 见 initializations.json，矩阵与噪声参数见 privacy_audit.json。"]
    lines += comparison_lines()
    upstream = read(BASE / "upstream_manifest.json")
    assert all(sha(ROOT / rel) == expected for rel, expected in upstream.items())
    save(BASE / "readonly_verification.json", dict(passed=True, files=len(upstream), time=now()))
    (BASE / "final_report.md").write_text("\n".join(lines) + "\n")
    save(BASE / "final_audit.json", dict(passed=True, completed=now(), full_trainings=90,
        selected_test_evaluations=30, pairing_verified=True, hyperparameters_sha256=sha(BASE / "best_hyperparameters.json")))


def running_report():
    (BASE / "final_report.md").write_text(
        "# Exp18c：已验收，等待/正在运行完整网格\n\n"
        "当前结果尚未完成，请查看 status.json 和 pipeline.log；本文件在全部测试评价后自动更新。\n\n"
        "采用 Exp18 CNN，逐字节复用 Exp18b 五份初始化。"
        "初始化文件与 SHA256 见 initializations.json，数据审计见 data_audit.json。\n\n"
        "固定 90 次完整训练；每个优化器根据两种噪声与五种子最终验证指标等权选择一个学习率，"
        "冻结后仅评价 30 个入选检查点的官方测试集，不重新训练。"
        "每次训练 ε=3、δ=1e-5、GDP 无采样放大，不代表 90 次组合预算 ε=3。\n")


def comparison_lines():
    import csv
    def table(path):
        with path.open() as handle:
            return list(csv.DictReader(handle))

    grids = {name: table(ROOT / name / "grid_summary.csv") for name in ("exp18b", "exp18c")}
    chosen = {name: read(ROOT / name / "best_hyperparameters.json")["groups"] for name in grids}
    per_seed = {name: table(ROOT / name / "formal_per_seed.csv") for name in grids}
    lines = ["", "## Exp18b 与 Exp18c 学习率扩展比较", "",
        "以下学习率搜索分析仅依据最终验证集。共同学习率由 IID/B2LT 各五种子等权平均选择；"
        "Exp18c 的正式选择范围严格为三个新候选。合并两轮候选的最优点仅用于判断搜索边界，不触发额外测试或训练。", "",
        "| 优化器 | 轮次 | LR | IID 验证准确率 | B2LT 验证准确率 | 两者等权平均 | 平均验证损失 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    candidates = {}
    for group, methods in GROUPS.items():
        candidates[group] = []
        for name, rows in grids.items():
            for lr in sorted({float(r["lr"]) for r in rows if r["group"] == group}):
                by_method = {r["method"]: r for r in rows if r["group"] == group and float(r["lr"]) == lr}
                iid, b2lt = [by_method[m] for m in methods]
                values = dict(group=group, experiment=name, lr=lr,
                    iid=float(iid["validation_accuracy_mean"]), b2lt=float(b2lt["validation_accuracy_mean"]),
                    iid_loss=float(iid["validation_loss_mean"]), b2lt_loss=float(b2lt["validation_loss_mean"]))
                values.update(average=(values["iid"] + values["b2lt"]) / 2,
                              average_loss=(values["iid_loss"] + values["b2lt_loss"]) / 2)
                candidates[group].append(values)
                lines.append(f"| {group} | {name} | {lr:g} | {values['iid']:.6f} | {values['b2lt']:.6f} | "
                             f"{values['average']:.6f} | {values['average_loss']:.6f} |")
    write_csv(BASE / "learning_rate_comparison.csv", [r for rows in candidates.values() for r in rows])
    lines += ["", "| 优化器 | Exp18b 共同 LR | Exp18c 最终共同 LR | Exp18c 边界 | 合并候选平均最优 LR | 合并候选边界 |",
              "|---|---:|---:|---|---:|---|"]
    def boundary(lr, values):
        return "最高边界" if lr == max(values) else "最低边界" if lr == min(values) else "内部"
    for group in GROUPS:
        rows = candidates[group]
        best = min(rows, key=lambda r: (-r["average"], r["average_loss"], r["lr"]))
        lr = chosen["exp18c"][group]["lr"]
        lines.append(f"| {group} | {chosen['exp18b'][group]['lr']:g} | {lr:g} | {boundary(lr, GRID[group])} | "
                     f"{best['lr']:g} | {boundary(best['lr'], [r['lr'] for r in rows])} |")
    lines += ["", "### IID、B2LT 与平均表现的边界分析", ""]
    for group, methods in GROUPS.items():
        rows = candidates[group]
        new = [r for r in rows if r["experiment"] == "exp18c"]
        for view, label in (("iid", "IID"), ("b2lt", "B2LT"), ("average", "两者平均")):
            best = min(rows, key=lambda r: (-r[view], r[view + "_loss"], r["lr"]))
            new_best = min(new, key=lambda r: (-r[view], r[view + "_loss"], r["lr"]))
            highest = max(rows, key=lambda r: r["lr"])
            edge = boundary(best["lr"], [r["lr"] for r in rows])
            implication = ("仍触及最高搜索边界，尚不能确认已进入稳定区域。" if edge == "最高边界" else
                "最高学习率已低于观测最优点，出现转折证据；五种子有限网格不能单凭此证明平坦稳定区域。" if edge == "内部" else
                "合并最优点处于最低边界，扩展方向未改善该视角的最优表现。")
            lines.append(f"- {group} / {label}：Exp18c 单独最优 LR={new_best['lr']:g}；两轮合并最优 LR={best['lr']:g}，"
                f"验证准确率 {best[view]:.6f}；最高候选 LR={highest['lr']:g} 时为 {highest[view]:.6f}。{implication}")
    lines += ["", "### 共同入选配置的五种子配对比较", "",
        "差值为 Exp18c − Exp18b，两轮各自按验证集共同选出的 LR；准确率差单位为百分点。"
        "平均视角先对同一种子的 IID/B2LT 取均值，再计算五种子统计量。测试结果只用于报告。", "",
        "| 优化器 | 视角 | 验证差均值（百分点） | 验证差 95% CI | 测试差均值（百分点） | 测试差 95% CI |",
        "|---|---|---:|---|---:|---|"]
    for group, methods in GROUPS.items():
        for view, label in (([methods[0]], "IID"), ([methods[1]], "B2LT"), (methods, "平均")):
            results = {}
            for metric in ("validation_accuracy", "test_accuracy"):
                by_experiment = {}
                for name, rows in per_seed.items():
                    by_experiment[name] = [np.mean([float(r[metric]) for r in rows
                        if r["group"] == group and r["method"] in view and int(r["seed"]) == seed]) for seed in SEEDS]
                results[metric] = stats(100 * (np.array(by_experiment["exp18c"]) - by_experiment["exp18b"]))
            v, t = results["validation_accuracy"], results["test_accuracy"]
            lines.append(f"| {group} | {label} | {v['mean']:+.4f} | [{v['ci95_low']:+.4f}, {v['ci95_high']:+.4f}] | "
                         f"{t['mean']:+.4f} | [{t['ci95_low']:+.4f}, {t['ci95_high']:+.4f}] |")
    lines += ["", "上述最优点和区间均为该固定网格的描述性结果；未对选参或多重比较进行校正。"
              "单独 IID/B2LT 最优点仅用于分析，最终正式配置仍遵循共同 LR 规则。"]
    return lines
