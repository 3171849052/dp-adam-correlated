"""Fixed protocol, immutable per-round plans, and pooled validation-only search."""
import os
from decimal import Decimal

from exp18c.continuation import BASE, ROOT
from exp18c.common import (CLIPS, GROUPS, PYTHON, SEEDS, WORKLOADS, digest,
    now, read, save, sha, tensor_hash, write_csv)

GRID = {"adam": [.08, .10, .15], "momentum": [1., 1.5, 2.], "stpex2": [.10, .15, .20]}


def next_rates(last):
    anchor = Decimal(str(last))
    exponent = anchor.adjusted()
    choices = sorted({Decimal(str(m)) * Decimal(10) ** e for e in range(exponent, exponent + 3)
                      for m in (1, 1.5, 2, 3, 4, 5, 6, 8)})
    return [float(x) for x in choices if x > anchor][:3]


def config(group, lr, method, seed, stage="grid", round_index=1):
    return dict(stage=stage, group=group, method=method, lr=lr, clip=CLIPS[group],
        eps=.005 if group == "stpex2" else 1e-8, seed=seed, steps=162,
        logical_batch=1000, physical_batch=1000, accumulation=1, precision="FP32",
        weight_decay=0, beta1=.9, beta2=.999, round=round_index)


def round_path(index):
    return BASE / "rounds" / f"{index:03d}"


def round_plan(index):
    value = read(round_path(index) / "plan.json")
    assert sha(round_path(index) / "plan.json") == read(round_path(index) / "plan_seal.json")["sha256"]
    return value


def plan(index=1):
    grid = round_plan(index)["grid"]
    return [config(g, lr, m, seed, round_index=index) for g, rates in grid.items()
            for lr in rates for seed in SEEDS for m in GROUPS[g]]


def smoke_plan(index):
    return [config(g, max(rates), m, SEEDS[0], "smoke", index)
            for g, rates in round_plan(index)["grid"].items() for m in GROUPS[g]]


def freeze_round(index, grid):
    value = dict(round=index, grid=grid, seeds=SEEDS, steps=162,
        protocol_sha256=sha(BASE / "protocol.json"), decision_source="pooled final validation only",
        formal_trainings=sum(len(v) * 10 for v in grid.values()))
    path = round_path(index) / "plan.json"
    if path.exists():
        assert round_plan(index) == value
    else:
        save(path, value)
        save(path.with_name("plan_seal.json"), dict(time=now(), sha256=sha(path)))
    plans = [dict(trial_id=trial_id(cfg), **cfg) for p in sorted((BASE / "rounds").glob("*/plan.json"))
             for cfg in plan(read(p)["round"])]
    write_csv(BASE / "trial_plan.csv", plans)
    return value


def trial_id(cfg):
    return digest(cfg)[:20]


def trial_dir(cfg):
    if cfg["stage"] == "smoke":
        return BASE / "smoke" / f"{cfg['round']:03d}" / trial_id(cfg)
    return BASE / "trials" / trial_id(cfg)


def validate_config(cfg):
    assert cfg == config(cfg["group"], cfg["lr"], cfg["method"], cfg["seed"], cfg["stage"], cfg["round"])
    assert cfg["lr"] in round_plan(cfg["round"])["grid"][cfg["group"]]
    assert cfg["seed"] in SEEDS and cfg["method"] in GROUPS[cfg["group"]]
    assert cfg["stage"] in ("grid", "smoke")
    if cfg["stage"] == "grid":
        accepted = round_path(cfg["round"]) / "acceptance.json"
        # Preparation/tests enumerate formal configs before round acceptance.
        if accepted.exists():
            assert read(accepted)["plan_sha256"] == sha(round_path(cfg["round"]) / "plan.json")


def env(gpu):
    result = os.environ.copy()
    result.update(CUDA_VISIBLE_DEVICES=read(BASE / "scheduling.json")["gpu_uuids"][str(gpu)],
        EXP18C_PHYSICAL_GPU=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        XDG_CACHE_HOME=str(BASE / "runtime/cache"), TORCH_HOME=str(BASE / "runtime/torch"), TMPDIR=str(BASE / "runtime"))
    return result


def load_matrix(name):
    import numpy as np
    path = BASE / "matrices" / f"{name}.npz"
    meta = read(path.with_suffix(".json"))
    assert sha(path) == meta["sha256"]
    with np.load(path) as arrays:
        return dict({key: arrays[key].copy() for key in arrays.files}, metadata=meta)


def code_seal():
    paths = list(BASE.glob("*.py")) + list((BASE / "tests").glob("*.py"))
    paths += [BASE / n for n in ("protocol.json", "scheduling.json", "assets.json", "initializations.json",
                                 "privacy_audit.json", "upstream_manifest.json")]
    paths += list((BASE / "matrices").glob("*")) + list((BASE / "initializations").glob("*.pt"))
    for folder in ("exp18", "exp18b", "exp18c"):
        paths += list((ROOT / folder).glob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths) if p.is_file()}


def verify_seal():
    assert read(BASE / "acceptance.json")["passed"]
    assert code_seal() == read(BASE / "acceptance.json")["code"]


def completed(cfg):
    directory = trial_dir(cfg)
    if not (directory / "complete.json").exists():
        return None
    assert read(directory / "config.json") == cfg
    for name, h in read(directory / "complete.json").items():
        assert sha(directory / name) == h
    row = read(directory / "summary.json")
    assert row["evaluation"] == "internal_validation"
    assert row["status"] == "diverged" or row["steps"] == 162
    return row
