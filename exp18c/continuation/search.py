"""Search decisions use validation summaries only; official test is isolated."""
import math
import numpy as np

from exp18c.continuation import BASE, ROOT
from exp18c.continuation.common import (GROUPS, SEEDS, read, save, sha, now, trial_id, trial_dir,
    round_path, round_plan, smoke_plan, write_csv)
from exp18b.report import stats


def verify_pairing(rows):
    pairs = {}
    for row in rows:
        cfg = row["config"]
        pairs.setdefault((cfg["group"], cfg["lr"], cfg["seed"]), []).append(row)
    for (group, lr, seed), pair in pairs.items():
        assert len(pair) == 2 and {r["config"]["method"] for r in pair} == set(GROUPS[group])
        assert pair[0]["pairing"] == pair[1]["pairing"]
        if all(r["status"] == "complete" for r in pair):
            assert pair[0]["final_innovation_rng_sha256"] == pair[1]["final_innovation_rng_sha256"]


def all_rows():
    rows = []
    for folder in (ROOT / "exp18b", ROOT / "exp18c", BASE):
        for path in sorted((folder / "trials").glob("*/summary.json")):
            if not path.with_name("complete.json").exists():
                continue
            row = read(path)
            assert row["evaluation"] == "internal_validation"
            row.update(source_directory=str(path.parent.relative_to(ROOT)),
                       source_experiment=str(folder.relative_to(ROOT)))
            rows.append(row)
    verify_pairing(rows)
    return rows


def aggregate(rows):
    points = []
    for group, methods in GROUPS.items():
        rates = sorted({r["config"]["lr"] for r in rows if r["config"]["group"] == group})
        for lr in rates:
            candidates = [r for r in rows if r["config"]["group"] == group and r["config"]["lr"] == lr]
            assert len(candidates) == 10
            for view, chosen in (("iid", methods[:1]), ("b2lt", methods[1:]), ("average", methods)):
                subset = [r for r in candidates if r["config"]["method"] in chosen]
                valid = all(r["status"] == "complete" and math.isfinite(r["metrics"]["loss"]) for r in subset)
                accuracy, loss = [], []
                if valid:
                    for seed in SEEDS:
                        seed_rows = [r for r in subset if r["config"]["seed"] == seed]
                        accuracy.append(sum(round(r["metrics"]["accuracy"] * 6000) for r in seed_rows) / (6000 * len(chosen)))
                        loss.append(float(np.mean([r["metrics"]["loss"] for r in seed_rows])))
                point = dict(group=group, view=view, lr=lr, valid=valid,
                    validation_accuracy=float(np.mean(accuracy)) if valid else None,
                    validation_loss=float(np.mean(loss)) if valid else None,
                    source_experiment=candidates[0]["source_experiment"],
                    diverged_trials=sum(r["status"] == "diverged" for r in subset))
                point["accuracy_stats"] = stats(accuracy) if valid else None
                point["loss_stats"] = stats(loss) if valid else None
                points.append(point)
    return points


def best_point(points):
    return min((p for p in points if p["valid"]),
               key=lambda p: (-p["validation_accuracy"], p["validation_loss"], p["lr"]))


def assess(points):
    policy = read(BASE / "protocol.json")["expansion"]
    result = {}
    for group in GROUPS:
        views = {}
        for view in ("iid", "b2lt", "average"):
            curve = sorted([p for p in points if p["group"] == group and p["view"] == view], key=lambda p: p["lr"])
            best = best_point(curve)
            tail = curve[-policy["confirmation_points"]:]
            lower = len(tail) == policy["confirmation_points"] and all(p["lr"] > best["lr"] and
                (not p["valid"] or p["validation_accuracy"] < best["validation_accuracy"]) for p in tail)
            drop = best["validation_accuracy"] - tail[-1]["validation_accuracy"] if tail[-1]["valid"] else None
            passed = lower and (drop is None or drop >= policy["min_accuracy_drop"])
            views[view] = dict(passed=passed, best_lr=best["lr"], best_accuracy=best["validation_accuracy"],
                highest_lr=curve[-1]["lr"], highest_accuracy=curve[-1]["validation_accuracy"],
                highest_drop=drop, witnesses=[dict(lr=p["lr"], accuracy=p["validation_accuracy"],
                    diverged_trials=p["diverged_trials"]) for p in tail])
        result[group] = dict(passed=all(v["passed"] for v in views.values()), views=views)
    return result


def publish(rows):
    points = aggregate(rows)
    turns = assess(points)
    flat = []
    for row in rows:
        cfg = row["config"]
        flat.append(dict(source_experiment=row["source_experiment"], source_directory=row["source_directory"],
            trial_id=trial_id(cfg), round=cfg.get("round", 0), group=cfg["group"], method=cfg["method"],
            lr=cfg["lr"], seed=cfg["seed"], clip=cfg["clip"], eps=cfg["eps"], status=row["status"],
            steps=row["steps"], validation_accuracy=row["metrics"]["accuracy"] if row["metrics"] else None,
            validation_loss=row["metrics"]["loss"] if row["metrics"] else None,
            train_seconds=row.get("train_seconds"), wall_seconds=row.get("seconds"), physical_gpu=row["physical_gpu"]))
    write_csv(BASE / "grid_per_seed.csv", flat)
    write_csv(BASE / "new_grid_per_seed.csv", [r for r in flat if r["source_experiment"] == str(BASE.relative_to(ROOT))])
    write_csv(BASE / "grid_summary.csv", [{k: v for k, v in p.items() if k not in ("accuracy_stats", "loss_stats")} for p in points])
    save(BASE / "turnover.json", dict(time=now(), all_passed=all(r["passed"] for r in turns.values()), groups=turns))
    return points, turns


def accept_round(index, execute):
    configs = smoke_plan(index)
    rows = execute(configs, "smoke")
    verify_pairing(rows)
    for row in rows:
        assert row["status"] == "diverged" or row["steps"] == 162
        cfg = row["config"]
        assert cfg["lr"] == max(round_plan(index)["grid"][cfg["group"]])
    ready = [read(trial_dir(c) / "ready.json") for c in configs]
    gpus = read(BASE / "scheduling.json")["physical_gpus"]
    assert len({r["physical_gpu"] for r in ready}) == min(len(gpus), len(configs))
    value = dict(passed=True, time=now(), smoke_trials=len(rows),
        smoke_divergences=sum(r["status"] == "diverged" for r in rows),
        plan_sha256=sha(round_path(index) / "plan.json"),
        physical_gpu_assignments=ready, pairing_verified=True,
        smoke_complete_hashes={str(trial_dir(c).relative_to(ROOT)): sha(trial_dir(c) / "complete.json") for c in configs})
    path = round_path(index) / "acceptance.json"
    if path.exists():
        assert read(path)["plan_sha256"] == value["plan_sha256"]
    else:
        save(path, value)
    return rows


def freeze_selection(rows, points, turns):
    assert all(r["passed"] for r in turns.values())
    groups = {g: dict(best_point([p for p in points if p["group"] == g and p["view"] == "average"]),
                clip=rows[0]["config"]["clip"], methods=GROUPS[g]) for g in GROUPS}
    selected = [r for r in rows if r["config"]["lr"] == groups[r["config"]["group"]]["lr"]]
    assert len(selected) == 30
    for g, value in groups.items():
        cfg = next(r["config"] for r in selected if r["config"]["group"] == g)
        value.update(clip=cfg["clip"], eps=cfg["eps"])
    value = dict(frozen=True, groups=groups, rule=read(BASE / "protocol.json")["selection"],
        official_test_used_for_selection=False, grid_trials=len(rows),
        protocol_sha256=sha(BASE / "protocol.json"),
        turnover=turns,
        selection_inputs={r["source_directory"]:sha(ROOT / r["source_directory"] / "summary.json") for r in rows},
        checkpoints={trial_id(r["config"]):dict(source_directory=r["source_directory"],
            checkpoint_sha256=sha(ROOT / r["source_directory"] / "checkpoint.pt")) for r in selected})
    path = BASE / "best_hyperparameters.json"
    if path.exists():
        assert read(path) == value
    else:
        save(path, value)
        save(BASE / "selection_freeze.json", dict(time=now(), sha256=sha(path)))
    return selected
