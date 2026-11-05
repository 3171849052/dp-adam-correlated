import json
import numpy as np

from exp18c.continuation import BASE, ROOT
from exp18c.continuation.common import GROUPS, SEEDS, now, read, save, sha, trial_id, write_csv
from exp18c.continuation.search import verify_pairing
from exp18b.report import stats


def report(selected, evaluations, points, turns):
    assert len(selected) == len(evaluations) == 30
    by_id = {r["trial_id"]:r for r in evaluations}
    per_seed = []
    for row in selected:
        cfg = row["config"]
        ev = by_id[trial_id(cfg)]
        directory = ROOT / row["source_directory"]
        assert ev["checkpoint_sha256"] == sha(directory / "checkpoint.pt")
        per_seed.append(dict(trial_id=trial_id(cfg), source_directory=row["source_directory"],
            group=cfg["group"], method=cfg["method"], seed=cfg["seed"], lr=cfg["lr"],
            clip=cfg["clip"], eps=cfg["eps"], validation_accuracy=row["metrics"]["accuracy"],
            validation_loss=row["metrics"]["loss"], test_accuracy=ev["metrics"]["accuracy"],
            test_loss=ev["metrics"]["loss"], train_seconds=row["train_seconds"], wall_seconds=row["seconds"],
            peak_memory_mib=row["peak_memory"] / 2**20, checkpoint_sha256=ev["checkpoint_sha256"],
            reused_official_test=ev.get("reused",False)))
    write_csv(BASE / "formal_per_seed.csv", per_seed)
    metrics = ("validation_accuracy", "validation_loss", "test_accuracy", "test_loss", "train_seconds", "wall_seconds", "peak_memory_mib")
    summaries, paired = [], []
    for group, methods in GROUPS.items():
        method_rows = {m:{r["seed"]:r for r in per_seed if r["method"] == m} for m in methods}
        for method in methods:
            summaries.append(dict(group=group, method=method, lr=method_rows[method][SEEDS[0]]["lr"],
                **{f"{metric}_{key}":v for metric in metrics for key,v in stats([method_rows[method][s][metric] for s in SEEDS]).items()}))
        for metric in metrics:
            differences = [method_rows[methods[1]][s][metric]-method_rows[methods[0]][s][metric] for s in SEEDS]
            paired.append(dict(group=group,metric=metric,direction="B2LT minus IID",
                **{f"seed_{s}":v for s,v in zip(SEEDS,differences)}, **stats(differences)))
    write_csv(BASE / "summary.csv", summaries)
    write_csv(BASE / "paired_differences.csv", paired)
    new_trials = list((BASE / "trials").glob("*/complete.json"))
    rounds = sorted((BASE / "rounds").glob("*/plan.json"))
    smoke_count = len(list((BASE / "smoke").glob("*/*/complete.json")))
    reused = sum(ev.get("reused",False) for ev in evaluations)
    lines = ["# Exp18c 继续搜索：所有优化器的高学习率转折", "", f"完成时间：{now()}。", "",
        f"完成 {len(rounds)} 轮扩展、{len(new_trials)} 次新增正式训练、{smoke_count} 次独立冒烟；"
        f"结合 Exp18b/Exp18c 原有 180 次训练，在全部候选中共同选参。30 个最终检查点完成官方测试评价，"
        f"其中 {reused} 个复用原有同一检查点的测试结果，{30-reused} 个进行新评价。", "",
        "实验参数严格沿用 Exp18b/Exp18c：MNIST 54000/6000/10000，Exp18 CNN 11,274 参数，"
        "复用五份初始化，3 epoch/162 步、物理 batch 1000、FP32/禁用 TF32。"
        "裁剪 Adam=3、Momentum=4、STP-Ex2=100；STP 稳定系数 0.005。优化器与三份 B2LT 矩阵不变。"
        "每个 LR 都是 IID/B2LT 五种子，初始权重、排列和创新流配对；仅搜索 LR。", "",
        "四张 GPU 自动检测编号与 UUID，按 UUID 绑定独立进程；每卡最多 12 个进程，保留 1536 MiB，"
        "每进程按 832 MiB 或更高冒烟实测预算安排。任务分配、利用率、显存与日志在 gpu_telemetry.csv、gpu_processes.jsonl、trials/。", "",
        "隐私为每次完整训练 ε=3、δ=1e-5，GDP 校准且无采样放大；多次运行的组合预算不等于 ε=3。"
        "验证集视为公开；训练损失和裁剪曲线仅为内部审计记录。", "",
        "转折判据在运行前冻结：每种优化器的 IID、B2LT、两者平均三条曲线，"
        "最高两个已测试 LR 都大于该视角最优 LR，且准确率均低于最优；最高 LR 至少下降 0.001（0.1 个百分点）。"
        "数值发散单独记录为失败端点，不虚构验证分数。此处判断的是观测到的高 LR 侧回落，"
        "不等价于证明全局最优、稳定平台或统计显著性。", "",
        "| 优化器 | 最终共同 LR | IID 验证准确率 | B2LT 验证准确率 | 等权平均 | 两种噪声均已越过峰值 |",
        "|---|---:|---:|---:|---:|---|"]
    for group, result in turns.items():
        assert result["passed"]
        lr = read(BASE / "best_hyperparameters.json")["groups"][group]["lr"]
        values = {p["view"]:p for p in points if p["group"] == group and p["lr"] == lr}
        lines.append(f"| {group} | {lr:g} | {values['iid']['validation_accuracy']:.6f} | "
            f"{values['b2lt']['validation_accuracy']:.6f} | {values['average']['validation_accuracy']:.6f} | 是 |")
    lines += ["", "单独 IID/B2LT 的最优 LR 仅用于曲线分析，正式配置继续使用两种噪声等权平均的共同 LR。", "",
        "| 优化器 | 视角 | 该视角最优 LR | 最优准确率 | 高端见证 LR | 对应准确率 | 最高 LR 回落（百分点） |",
        "|---|---|---:|---:|---|---|---:|"]
    for group, result in turns.items():
        for view, value in result["views"].items():
            witness = value["witnesses"]
            rates = ", ".join(format(w["lr"], "g") for w in witness)
            scores = ", ".join("发散" if w["accuracy"] is None else f"{w['accuracy']:.6f}" for w in witness)
            drop = "数值发散" if value["highest_drop"] is None else f"{100*value['highest_drop']:.4f}"
            lines.append(f"| {group} | {view} | {value['best_lr']:g} | {value['best_accuracy']:.6f} | "
                f"{rates} | {scores} | {drop} |")
    lines += ["", "## 全部验证曲线与轮次", "",
        "| 优化器 | LR | 来源 | IID | B2LT | 两者平均 |", "|---|---:|---|---:|---:|---:|"]
    for group in GROUPS:
        for lr in sorted({p["lr"] for p in points if p["group"] == group}):
            values = {p["view"]:p for p in points if p["group"] == group and p["lr"] == lr}
            scores = ["发散" if not values[v]["valid"] else f"{values[v]['validation_accuracy']:.6f}" for v in ("iid","b2lt","average")]
            lines.append(f"| {group} | {lr:g} | {values['average']['source_experiment']} | {' | '.join(scores)} |")
    lines += ["", "各轮不可变计划和冒烟验收见 rounds/。未使用官方测试集决定继续、停止或选择 LR；"
        "搜索全部结束后才冻结 best_hyperparameters.json 和 selection_freeze.json。", "",
        "## 最终官方测试", "", "五种子均值、样本标准差和 Student-t 95% CI（df=4），未校正选参或多重比较。", "",
        "| 方法 | 共同 LR | 测试准确率 | SD | 95% CI | 测试损失 |", "|---|---:|---:|---:|---|---:|"]
    for r in summaries:
        lines.append(f"| {r['method']} | {r['lr']:g} | {r['test_accuracy_mean']:.6f} | {r['test_accuracy_sd']:.6f} | "
            f"[{r['test_accuracy_ci95_low']:.6f}, {r['test_accuracy_ci95_high']:.6f}] | {r['test_loss_mean']:.6f} |")
    lines += ["", "逐种子验证、测试、耗时、显存及来源见 formal_per_seed.csv；完整统计见 summary.csv；"
        "B2LT − IID 配对差异见 paired_differences.csv。复用旧结果的路径与原始测试时间保留在 official_test/。", "",
        "Exp18、Exp18b、Exp18c 原有文件及 data/ 保持只读，文件哈希核验见 readonly_verification.json。"]
    (BASE / "final_report.md").write_text("\n".join(lines)+"\n")
    return dict(rounds=len(rounds), new_formal_trainings=len(new_trials), smoke_trainings=smoke_count,
        selected_test_evaluations=30, reused_official_tests=reused, new_official_tests=30-reused)
