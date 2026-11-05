"""Detached launcher. Transfer a held flock to the background session without a race."""
import fcntl
import os
import subprocess

from exp18c import BASE, ROOT
from exp18c.common import PYTHON, env, now, read, save, verify_seal


def main():
    verify_seal()
    with (BASE / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [PYTHON, "-B", "-m", "exp18c.pipeline"]
        log = BASE / "pipeline.log"
        policy = read(BASE / "scheduling.json")
        environment = os.environ.copy()
        environment.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(BASE / "runtime"))
        environment["EXP18C_LOCK_FD"] = str(lock.fileno())
        with log.open("a") as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        save(BASE / "background.json", dict(pid=process.pid, launched=now(), command=command,
            cwd=str(ROOT), log=str(log), detached_session=True, stdin="/dev/null",
            physical_gpus=policy["physical_gpus"], max_processes_per_gpu=policy["target_per_gpu"],
            scheduling=read(BASE / "scheduling.json"),
            resume_command=f"{PYTHON} -B -m exp18c.start_background",
            recovery="Only completed hash-verified trials are reused; incomplete trials preserved for inspection"))
    print(process.pid, flush=True)


if __name__ == "__main__":
    main()
