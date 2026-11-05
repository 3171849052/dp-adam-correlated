import json
import math
import subprocess

from exp18c import BASE, ROOT
from exp18c.common import (GROUPS, GRID, PYTHON, SEEDS, code_seal, config, now, read, save,
                           sha, trial_dir)
from exp18c.report import running_report, verify_pairing
from exp18c.scheduler import execute
from exp18c.setup import prepare


def main():
    prepare()
    with (BASE / "unit_tests.log").open("w") as handle:
        subprocess.run([PYTHON, "-B", "-m", "unittest", "discover", "-s", "exp18c/tests", "-v"],
                       cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    configs = [config(g, GRID[g][-1], m, SEEDS[0], "smoke") for g, methods in GROUPS.items() for m in methods]
    rows = execute(configs, "smoke")
    verify_pairing(rows)
    ready = [read(trial_dir(cfg) / "ready.json") for cfg in configs]
    assert len(ready) == 6 and all(r["cuda_memory"] > 0 for r in ready)
    gpus = read(BASE / "scheduling.json")["physical_gpus"]
    assert {r["physical_gpu"] for r in ready} == set(gpus)
    assert sorted(sum(r["physical_gpu"] == g for r in ready) for g in gpus) == [1, 1, 2, 2]
    assert read(BASE / "smoke_status.json")["all_gpu_overlap"]
    for cfg in configs:
        records = [json.loads(line) for line in (trial_dir(cfg) / "steps.jsonl").read_text().splitlines()]
        assert len(records) == 162 and [r["step"] for r in records] == list(range(1, 163))
        assert all(math.isfinite(r["loss"]) for r in records)
        if cfg["group"] == "stpex2":
            assert all(0 <= r["alpha"] <= 1 for r in records)
    policy = read(BASE / "scheduling.json")
    measured = read(BASE / "smoke_status.json")["peak_worker_nvml_mib"]
    # Retain Exp18's audited reservation unless the new physical measurement needs more.
    policy["worker_budget_mib"] = max(policy["worker_budget_mib"], math.ceil(measured * 1.1 / 64) * 64)
    policy["smoke_peak_worker_nvml_mib"] = measured
    save(BASE / "scheduling.json", policy)
    accepted = dict(passed=True, time=now(), smoke_trials=6, smoke_steps_per_trial=162,
        physical_batch=1000, logical_batch=1000, accumulation=1, precision="FP32", physical_gpus=gpus,
        smoke_workers_per_gpu={g: sum(r["physical_gpu"] == g for r in ready) for g in gpus}, pairing_verified=True, matrix_and_privacy="privacy_audit.json",
        tests="unit_tests.log", initialized_from="Exact Exp18b initialization file copies",
        code=code_seal(), smoke_complete_hashes={str(trial_dir(c).relative_to(ROOT)): sha(trial_dir(c) / "complete.json") for c in configs})
    save(BASE / "acceptance.json", accepted)
    running_report()
    print("Acceptance passed; code/configuration/initialization/matrix hashes frozen", flush=True)


if __name__ == "__main__":
    main()
