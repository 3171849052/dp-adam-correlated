"""Exp18 admission policy and round-robin subprocess scheduling on physical GPUs."""
import csv
import json
import subprocess
import time

from exp18.scheduler import admission_limit
from exp18c.continuation import BASE, ROOT
from exp18c.continuation.common import (PYTHON, completed, env, now, read, save, trial_dir, trial_id,
                           validate_config)


def telemetry():
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                         "--format=csv,noheader,nounits"], text=True)
    metrics = {}
    for line in gpu.splitlines():
        index, uuid, total, used, util = [field.strip() for field in line.split(",")]
        metrics[int(index)] = dict(uuid=uuid, total=int(total), used=int(used), utilization=int(util))
    resident = {}
    for line in processes.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[1].isdigit() and fields[2].isdigit():
            resident[int(fields[1])] = int(fields[2])
    return metrics, resident, processes


def execute(configs, stage):
    policy = read(BASE / "scheduling.json")
    if stage == "smoke":
        policy = dict(policy, target_per_gpu=2)
    gpus = policy["physical_gpus"]
    pending, rows, active, failures = [], [], [], []
    for cfg in configs:
        validate_config(cfg)
        old = completed(cfg)
        if old:
            rows.append(old)
        else:
            path = trial_dir(cfg)
            assert not path.exists(), f"Incomplete trial preserved at {path}; inspect before retrying"
            pending.append(cfg)
    capacity = {gpu: policy["target_per_gpu"] for gpu in gpus}
    last, overlap, peak_resident = 0., False, 0
    telemetry_path = BASE / "gpu_telemetry.csv"
    is_new = not telemetry_path.exists()
    with telemetry_path.open("a", newline="", buffering=1) as handle, \
            (BASE / "gpu_processes.jsonl").open("a", buffering=1) as process_log:
        writer = csv.DictWriter(handle, fieldnames=["time", "stage", "physical_gpu", "gpu_uuid",
             "memory_total_mib", "memory_used_mib", "gpu_utilization_percent", "own_resident_mib",
             "active_workers", "admission_capacity"])
        if is_new:
            writer.writeheader()
        while pending or active:
            for item in list(active):
                process, cfg, gpu, logfile = item
                if process.poll() is None:
                    continue
                logfile.close()
                active.remove(item)
                if process.returncode:
                    error = dict(trial_id=trial_id(cfg), returncode=process.returncode,
                                 physical_gpu=gpu, time=now())
                    if (trial_dir(cfg) / "failure.json").exists():
                        error["detail"] = read(trial_dir(cfg) / "failure.json")
                    else:
                        save(trial_dir(cfg) / "failure.json", error)
                    failures.append(error)
                else:
                    row = completed(cfg)
                    assert row
                    rows.append(row)
            sampled = time.monotonic() - last >= policy["telemetry_seconds"] or last == 0
            if sampled:
                metrics, resident, process_text = telemetry()
                process_log.write(json.dumps(dict(time=now(), stage=stage, processes=process_text)) + "\n")
                last = time.monotonic()
                own = {g: sum(resident.get(p.pid, 0) for p, c, gpu, f in active if gpu == g) for g in gpus}
                peak_resident = max([peak_resident] + [resident.get(p.pid, 0) for p, c, g, f in active])
                for gpu in gpus:
                    m = metrics[gpu]
                    limit = admission_limit(m["total"], m["used"], own[gpu], policy)
                    if limit < capacity[gpu]:
                        capacity[gpu] = limit
                    elif limit > capacity[gpu] and m["utilization"] < 90:
                        capacity[gpu] = min(limit, capacity[gpu] + 1)
            if not failures:
                while pending:
                    choices = [g for g in gpus if sum(a[2] == g for a in active) < capacity[g]]
                    if not choices:
                        break
                    gpu = min(choices, key=lambda g: sum(a[2] == g for a in active))
                    cfg = pending.pop(0)
                    path = trial_dir(cfg)
                    save(path / "config.json", cfg)
                    logfile = (path / "worker.log").open("w")
                    process = subprocess.Popen([PYTHON, "-B", "-m", "exp18c.continuation.train", str(path / "config.json")],
                        cwd=ROOT, env=env(gpu), stdin=subprocess.DEVNULL, stdout=logfile,
                        stderr=subprocess.STDOUT)
                    active.append((process, cfg, gpu, logfile))
            overlap |= all(sum(a[2] == g for a in active) >= 1 for g in gpus)
            state = dict(status="draining_after_failure" if failures and active else
                         "failed" if failures else "running" if pending or active else "complete",
                         stage=stage, pending=len(pending), completed=len(rows), total=len(configs),
                         active=[dict(pid=p.pid, trial_id=trial_id(c), physical_gpu=g) for p, c, g, f in active],
                         capacity=capacity, failures=failures, all_gpu_overlap=overlap,
                         peak_worker_nvml_mib=peak_resident, time=now())
            save(BASE / f"{stage}_status.json", state)
            save(BASE / "status.json", state)
            if sampled:
                for gpu in gpus:
                    m = metrics[gpu]
                    writer.writerow(dict(time=now(), stage=stage, physical_gpu=gpu, gpu_uuid=m["uuid"],
                        memory_total_mib=m["total"], memory_used_mib=m["used"],
                        gpu_utilization_percent=m["utilization"], own_resident_mib=own[gpu],
                        active_workers=sum(a[2] == gpu for a in active), admission_capacity=capacity[gpu]))
            if failures and not active:
                break
            if active or pending:
                time.sleep(0.5)
    if failures:
        raise RuntimeError(f"{len(failures)} trial(s) failed; stopped dispatch, active workers drained")
    assert len(rows) == len(configs)
    return rows
