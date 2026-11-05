import csv
import fcntl
import os
import signal
import subprocess
import time
import traceback

from exp18c import BASE, ROOT
from exp18c.common import PYTHON, env, now, plan, read, save, sha, trial_dir, trial_id, verify_seal
from exp18c.report import report, select
from exp18c.scheduler import execute, telemetry


def evaluate_selected(selected):
    queue, rows, active, failures = [], [], [], []
    selection_hash = sha(BASE / "best_hyperparameters.json")
    for row in selected:
        cfg = row["config"]
        intent = dict(trial_id=trial_id(cfg), checkpoint_sha256=sha(trial_dir(cfg) / "checkpoint.pt"),
                      selection_sha256=selection_hash)
        out = BASE / "official_test" / trial_id(cfg)
        if (out / "complete.json").exists():
            assert read(out / "intent.json") == intent
            for name, h in read(out / "complete.json").items():
                assert sha(out / name) == h
            result = read(out / "result.json")
            assert result["metrics"]["n"] == 10000 and result["selection_sha256"] == selection_hash
            rows.append(result)
        else:
            assert not out.exists(), f"Incomplete official test preserved at {out}; inspect before any repeat"
            queue.append((out, intent))
    last_sample = 0.
    while queue or active:
        for item in list(active):
            process, out, gpu, logfile = item
            if process.poll() is None:
                continue
            logfile.close()
            active.remove(item)
            if process.returncode:
                failures.append(dict(trial_id=out.name, returncode=process.returncode))
            else:
                assert (out / "complete.json").exists()
                rows.append(read(out / "result.json"))
        if not failures:
            for gpu in read(BASE / "scheduling.json")["physical_gpus"]:
                if not queue or any(a[2] == gpu for a in active):
                    continue
                out, intent = queue.pop(0)
                save(out / "intent.json", intent)
                logfile = (out / "worker.log").open("w")
                process = subprocess.Popen([PYTHON, "-B", "-m", "exp18c.evaluate", str(out)],
                    cwd=ROOT, env=env(gpu), stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT)
                active.append((process, out, gpu, logfile))
        if time.monotonic() - last_sample >= 2:
            metrics, resident, process_text = telemetry()
            policy = read(BASE / "scheduling.json")
            with (BASE / "gpu_telemetry.csv").open("a", newline="") as handle:
                writer = csv.writer(handle)
                for gpu in policy["physical_gpus"]:
                    m = metrics[gpu]
                    own = sum(resident.get(p.pid, 0) for p, d, g, f in active if g == gpu)
                    writer.writerow([now(), "official_test", gpu, m["uuid"], m["total"], m["used"],
                        m["utilization"], own, sum(g == gpu for p, d, g, f in active), 1])
            with (BASE / "gpu_processes.jsonl").open("a") as handle:
                import json
                handle.write(json.dumps(dict(time=now(), stage="official_test", processes=process_text)) + "\n")
            last_sample = time.monotonic()
        save(BASE / "status.json", dict(status="running" if not failures else "failed", stage="official_test",
            total=30, completed=len(rows), pending=len(queue), failures=failures, time=now(),
            active=[dict(pid=p.pid, trial_id=d.name, physical_gpu=g) for p, d, g, f in active]))
        if failures and not active:
            break
        if active or queue:
            time.sleep(0.5)
    assert not failures, f"Official test evaluation failed: {failures}"
    return rows


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    inherited_fd = os.environ.pop("EXP18C_LOCK_FD", None)
    lock = os.fdopen(int(inherited_fd), "a") if inherited_fd else (BASE / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        verify_seal()
        save(BASE / "pipeline_process.json", dict(pid=os.getpid(), started=now(), session_id=os.getsid(0)))
        print(f"{now()} Starting/resuming Exp18c full 90-trial grid", flush=True)
        rows = execute(plan(), "grid")
        selected = select(rows)
        assert len(selected) == 30
        print(f"{now()} Validation selection frozen; evaluating 30 selected checkpoints", flush=True)
        evaluations = evaluate_selected(selected)
        report(selected, evaluations)
        verify_seal()
        save(BASE / "status.json", dict(status="complete", stage="complete", full_trainings=90,
            selected_test_evaluations=30, active=[], finished=now()))
        print(f"{now()} Exp18c complete", flush=True)
    except Exception as error:
        failure = dict(status="failed", time=now(), type=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc())
        save(BASE / "pipeline_failure.json", failure)
        save(BASE / "status.json", {**failure, "last_status": read(BASE / "status.json") if (BASE / "status.json").exists() else None})
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
