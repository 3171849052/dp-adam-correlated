import csv
import fcntl
import os
import signal
import subprocess
import time
import traceback

from exp18c.continuation import BASE, ROOT
from exp18c.continuation.common import (PYTHON, env, now, plan, read, save, sha, trial_dir, trial_id,
    verify_seal, freeze_round, round_path, round_plan, next_rates)
from exp18c.continuation.search import all_rows, aggregate, assess, publish, accept_round, freeze_selection
from exp18c.continuation.scheduler import execute, telemetry
from exp18c.continuation.report import report


def evaluate_selected(selected):
    queue, rows, active, failures = [], [], [], []
    selection_hash = sha(BASE / "best_hyperparameters.json")
    for row in selected:
        cfg = row["config"]
        intent = dict(trial_id=trial_id(cfg), checkpoint_sha256=sha(ROOT / row["source_directory"] / "checkpoint.pt"),
                      selection_sha256=selection_hash, source_directory=row["source_directory"])
        out = BASE / "official_test" / trial_id(cfg)
        if (out / "complete.json").exists():
            assert read(out / "intent.json") == intent
            for name, h in read(out / "complete.json").items():
                assert sha(out / name) == h
            result = read(out / "result.json")
            assert result["metrics"]["n"] == 10000 and result["selection_sha256"] == selection_hash
            rows.append(result)
        else:
            source = ROOT / row["source_experiment"] / "official_test" / trial_id(cfg)
            if (source / "complete.json").exists():
                for name, expected in read(source / "complete.json").items():
                    assert sha(source / name) == expected
                previous = read(source / "result.json")
                assert previous["checkpoint_sha256"] == intent["checkpoint_sha256"]
                assert previous["metrics"]["n"] == 10000
                result = dict(previous, **intent, reused=True, reused_from=str(source.relative_to(ROOT)),
                    original_evaluation_time=previous["time"], recorded=now())
                save(out / "intent.json", intent)
                save(out / "result.json", result)
                save(out / "complete.json", {p.name:sha(p) for p in out.iterdir() if p.is_file()})
                rows.append(result)
                continue
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
                process = subprocess.Popen([PYTHON, "-B", "-m", "exp18c.continuation.evaluate", str(out)],
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
    inherited_fd = os.environ.pop("EXP18C_CONTINUATION_LOCK_FD", None)
    lock = os.fdopen(int(inherited_fd), "a") if inherited_fd else (BASE / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        verify_seal()
        save(BASE / "pipeline_process.json", dict(pid=os.getpid(), started=now(), session_id=os.getsid(0)))
        index = 1
        while True:
            print(f"{now()} Round {index}: {round_plan(index)['grid']}", flush=True)
            accept_round(index, execute)
            verify_seal()
            rows = execute(plan(index), "grid")
            save(round_path(index) / "complete.json", dict(time=now(), formal_trials=len(rows),
                plan_sha256=sha(round_path(index) / "plan.json")))
            pooled = all_rows()
            points, turns = publish(pooled)
            print(f"{now()} Round {index} turnover: { {g:v['passed'] for g,v in turns.items()} }", flush=True)
            if all(v["passed"] for v in turns.values()):
                break
            grid = {g:next_rates(max(p["lr"] for p in points if p["group"] == g))
                    for g,v in turns.items() if not v["passed"]}
            index += 1
            freeze_round(index, grid)
        selected = freeze_selection(pooled, points, turns)
        print(f"{now()} All nine validation views turned over; selection frozen", flush=True)
        evaluations = evaluate_selected(selected)
        counts = report(selected, evaluations, points, turns)
        manifest = read(BASE / "upstream_manifest.json")
        assert all(sha(ROOT / rel) == expected for rel,expected in manifest.items())
        save(BASE / "readonly_verification.json", dict(passed=True,time=now(),files=len(manifest)))
        verify_seal()
        save(BASE / "final_audit.json", dict(passed=True,time=now(),all_nine_turnovers=True,**counts))
        save(BASE / "status.json", dict(status="complete",stage="complete",active=[],finished=now(),**counts))
        print(f"{now()} Continuation complete: {counts}", flush=True)
    except Exception as error:
        failure = dict(status="failed",time=now(),type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        save(BASE / "pipeline_failure.json", failure)
        save(BASE / "status.json", failure)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
