"""Prepare only local assets and audit the unchanged Exp18 algorithms/matrices."""
import shutil
import sys
import subprocess

import numpy as np
import torch

from exp18.model import CNN
from exp18.matrices import audit, workload
from exp18.privacy import fixed_epoch_sensitivity, target_mu, epsilon_from_mu
from exp18c import BASE, DATA, ROOT
from exp18c.common import (CLIPS, GROUPS, SEEDS, WORKLOADS, load_matrix, now, plan,
                           read, save, sha, tensor_hash, trial_id, write_csv)


def prepare():
    assert "envs/curve/" in sys.executable, "Use Conda curve"
    assert not (BASE / "acceptance.json").exists(), "Already frozen; use resume launcher"
    torch.set_num_threads(2)
    accepted = read(ROOT / "exp18b/acceptance.json")["code"]
    for relative, expected in accepted.items():
        assert sha(ROOT / relative) == expected, f"Exp18b frozen source changed: {relative}"
    upstream = {str(p.relative_to(ROOT)): sha(p) for folder in ("exp18", "exp18b", "data")
                for p in sorted((ROOT / folder).rglob("*")) if p.is_file()}
    save(BASE / "upstream_manifest.json", upstream)
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits"], text=True)
    inventory = [dict(zip(("index", "uuid", "name", "memory_total_mib"),
                        [f.strip() for f in line.split(",")])) for line in devices.splitlines()]
    assert len(inventory) == 4, "Expected four available physical GPUs"
    policy = read(ROOT / "exp18b/scheduling.json")
    policy.update(physical_gpus=[int(g["index"]) for g in inventory],
                  gpu_uuids={g["index"]: g["uuid"] for g in inventory}, inventory=inventory,
                  discovery_time=now())
    policy.pop("smoke_peak_worker_nvml_mib", None)
    save(BASE / "scheduling.json", policy)
    reused = ["model.py", "optimizer.py", "noise.py", "privacy.py", "matrices.py", "scheduler.py"]
    for name in reused:
        path = ROOT / "exp18" / name
        assert sha(path) == accepted[str(path.relative_to(ROOT))], f"Exp18 version changed: {name}"
    assert sha(ROOT / "stp-ex2.pdf") == accepted["stp-ex2.pdf"]

    assets = read(ROOT / "exp18b/assets.json")
    for relative, expected in assets.items():
        assert sha(ROOT / relative) == expected, f"Data changed: {relative}"
    x = torch.load(DATA / "train.pt", map_location="cpu", weights_only=True)
    test = torch.load(DATA / "test.pt", map_location="cpu", weights_only=True)
    for data, count in ((x, 60000), (test, 10000)):
        assert data["images"].shape == (count, 1, 28, 28)
        assert data["images"].dtype == torch.float32 and torch.isfinite(data["images"]).all()
        assert data["labels"].shape == (count,) and data["labels"].dtype == torch.int64
        assert int(data["labels"].min()) == 0 and int(data["labels"].max()) == 9
    with np.load(DATA / "split.npz") as split:
        assert split["train"].shape == (54000,) and split["valid"].shape == (6000,)
        assert np.array_equal(np.sort(np.r_[split["train"], split["valid"]]), np.arange(60000))
    save(BASE / "assets.json", assets)
    save(BASE / "data_audit.json", dict(passed=True, time=now(), no_download=True,
         split_sizes=[54000, 6000, 10000], official_test_evaluated=False,
         files=[dict(path=str(ROOT / rel), sha256=h) for rel, h in assets.items()]))

    initializations = {}
    for seed in SEEDS:
        original = read(ROOT / "exp18b/initializations.json")[str(seed)]
        source = ROOT / original["path"]
        assert sha(source) == original["sha256"]
        path = BASE / "initializations" / source.name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        assert sha(path) == original["sha256"]
        assert tensor_hash(torch.load(path, weights_only=True)) == original["tensor_sha256"]
        initializations[str(seed)] = dict(original, path=str(path.relative_to(ROOT)),
            source=original["path"], source_sha256=original["sha256"])
    save(BASE / "initializations.json", initializations)
    rows = [dict(trial_id=trial_id(cfg), **cfg) for cfg in plan()]
    assert len(rows) == 90 and len({r["trial_id"] for r in rows}) == 90
    write_csv(BASE / "trial_plan.csv", rows)

    matrix_audits = {}
    for name in WORKLOADS.values():
        for suffix in ("npz", "json"):
            source = ROOT / "exp18b/matrices" / f"{name}.{suffix}"
            assert sha(source) == accepted[str(source.relative_to(ROOT))]
            dest = BASE / "matrices" / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        m = load_matrix(name)
        meta = m["metadata"]
        assert {k: meta[k] for k in ("T", "K", "P", "Q")} == dict(T=162, K=54, P=54, Q=5)
        assert meta["selected"] == "b2lt" and meta["first_order_pass"]
        assert meta["projected_gradient_inf"] <= 1e-7
        np.testing.assert_allclose(m["A"], workload(name), atol=1e-12)
        theta, omega = np.array(meta["theta"]), np.array(meta["omega"])
        assert len(theta) == len(omega) == 5 and np.all((theta > 0) & (theta < 1))
        assert np.all(np.diff(theta) > 0) and np.all(omega < 0)
        assert 1 + omega.sum() > 0 and 1 + (omega / (1 - theta)).sum() > 0
        d = np.r_[1., (omega[:, None] * theta[:, None] ** np.arange(53)).sum(0), np.zeros(108)]
        np.testing.assert_allclose(m["d"], d, atol=1e-14)
        lag = np.arange(162)[:, None] - np.arange(162)[None, :]
        np.testing.assert_allclose(m["D"], np.where(lag >= 0, d[lag.clip(0)], 0), atol=1e-14)
        result = audit(m["A"], m["C"], m["D"])
        matrix_audits[name] = dict(**result, sha256=meta["sha256"],
                                 source=str(ROOT / "exp18b/matrices" / f"{name}.npz"))

    privacy = {}
    mu = target_mu(3, 1e-5)
    for group, methods in GROUPS.items():
        for method in methods:
            iid = method.endswith("-iid")
            C = np.eye(162) if iid else load_matrix(WORKLOADS[group])["C"]
            sensitivity = fixed_epoch_sensitivity(C, 3, 54)
            sigma = CLIPS[group] * sensitivity / mu
            epsilon = epsilon_from_mu(CLIPS[group] * sensitivity / sigma, 1e-5)
            assert abs(epsilon - 3) < 1e-9
            privacy[method] = dict(epsilon=epsilon, delta=1e-5, calibration="GDP",
                mu=mu, clip=CLIPS[group], sensitivity=sensitivity, sigma_sum=sigma,
                sigma_average=sigma / 1000, adjacency="add/remove with fixed zeroed slot",
                sampling_amplification=False, participations=3, participation_interval=54,
                T=162, K=54, P=54 if not iid else 1, Q=5 if not iid else 0,
                matrix_sha256=None if iid else matrix_audits[WORKLOADS[group]]["sha256"],
                scope="one complete training, not the composition of 90 trainings")
    assert privacy == read(ROOT / "exp18b/privacy_audit.json")["methods"]
    save(BASE / "privacy_audit.json", dict(passed=True, methods=privacy, matrices=matrix_audits,
        algorithm_sources={"exp18/" + name: sha(ROOT / "exp18" / name) for name in reused},
        stpex2_pdf_sha256=sha(ROOT / "stp-ex2.pdf"), composed_epsilon_is_not_3=True))
    save(BASE / "preflight.json", dict(passed=True, time=now(), environment=sys.executable,
        data="passed", model="Exp18 CNN, byte-for-byte Exp18b initializations",
        cache_weights_used=False, protocol_revision=read(BASE / "protocol.json")["authorization_revision"],
        upstream_readonly_manifest="upstream_manifest.json", matrices="passed", privacy="passed",
        planned_full_trainings=90))
    print("Local data, five initializations, 90-trial plan, matrices and GDP calibration passed", flush=True)


if __name__ == "__main__":
    prepare()
