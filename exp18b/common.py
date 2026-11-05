import csv
import hashlib
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from exp18b import BASE, ROOT
from exp18.common import GROUPS, PYTHON, digest, read, save, sha

SEEDS = [20261111, 20261112, 20261113, 20261114, 20261115]
WORKLOADS = {"adam": "sgd", "momentum": "momentum", "stpex2": "momentum-bias"}
GRID = {"adam": [0.02, 0.03, 0.04], "momentum": [0.20, 0.30, 0.40],
        "stpex2": [0.02, 0.03, 0.04]}
CLIPS = {"adam": 3, "momentum": 4, "stpex2": 100}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_csv(path, rows):
    assert rows
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def config(group, lr, method, seed, stage="grid"):
    return dict(stage=stage, group=group, method=method, lr=lr, clip=CLIPS[group],
                eps=0.005 if group == "stpex2" else 1e-8, seed=seed,
                steps=56 if stage == "smoke" else 162, logical_batch=1000,
                physical_batch=1000, accumulation=1, precision="FP32",
                weight_decay=0, beta1=0.9, beta2=0.999)


def plan():
    return [config(group, lr, method, seed) for group, methods in GROUPS.items()
            for lr in GRID[group] for seed in SEEDS for method in methods]


def trial_id(cfg):
    return digest(cfg)[:20]


def trial_dir(cfg):
    return BASE / ("smoke" if cfg["stage"] == "smoke" else "trials") / trial_id(cfg)


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def env(gpu):
    result = os.environ.copy()
    result.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                  PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  OPENBLAS_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                  XDG_CACHE_HOME=str(BASE / "runtime/cache"),
                  TORCH_HOME=str(BASE / "runtime/torch"), TMPDIR=str(BASE / "runtime"))
    return result


def code_seal():
    paths = (list(BASE.glob("*.py")) + list((BASE / "tests").glob("*.py"))
             + [BASE / name for name in ("protocol.json", "scheduling.json", "trial_plan.csv",
                                        "assets.json", "initializations.json", "privacy_audit.json")]
             + list((BASE / "matrices").glob("*"))
             + list((BASE / "initializations").glob("*.pt"))
             + list((ROOT / "exp18").glob("*.py")) + [ROOT / "stp-ex2.pdf"])
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths) if path.is_file()}


def verify_seal():
    accepted = read(BASE / "acceptance.json")
    assert accepted["passed"]
    assert code_seal() == accepted["code"], "Frozen code/configuration changed"


def load_matrix(name):
    import numpy as np
    path = BASE / "matrices" / f"{name}.npz"
    meta = read(path.with_suffix(".json"))
    assert sha(path) == meta["sha256"]
    with np.load(path) as arrays:
        return dict({key: arrays[key].copy() for key in arrays.files}, metadata=meta)


def validate_config(cfg):
    assert cfg == config(cfg["group"], cfg["lr"], cfg["method"], cfg["seed"], cfg["stage"])
    assert cfg["seed"] in SEEDS and cfg["method"] in GROUPS[cfg["group"]]
    assert cfg["lr"] in GRID[cfg["group"]] and cfg["stage"] in ("grid", "smoke")


def completed(cfg):
    directory = trial_dir(cfg)
    if not (directory / "complete.json").exists():
        return None
    assert read(directory / "config.json") == cfg
    for name, expected in read(directory / "complete.json").items():
        assert sha(directory / name) == expected, f"Artifact changed: {directory / name}"
    summary = read(directory / "summary.json")
    assert summary["steps"] == cfg["steps"] and summary["evaluation"] == "internal_validation"
    return summary
