"""Detached launcher. Transfer a held flock to the background session without a race."""
import fcntl
import os
import subprocess

from exp18b import BASE, ROOT
from exp18b.common import PYTHON, env, now, read, save, verify_seal


def main():
    verify_seal()
    with (BASE / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [PYTHON, "-B", "-m", "exp18b.pipeline"]
        log = BASE / "pipeline.log"
        environment = env("1,2,3")
        environment["EXP18B_LOCK_FD"] = str(lock.fileno())
        with log.open("a") as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        save(BASE / "background.json", dict(pid=process.pid, launched=now(), command=command,
            cwd=str(ROOT), log=str(log), detached_session=True, stdin="/dev/null",
            physical_gpus=[1, 2, 3], max_processes_per_gpu=12,
            scheduling=read(BASE / "scheduling.json"),
            resume_command=f"{PYTHON} -B -m exp18b.start_background",
            recovery_command=f"{PYTHON} -B -m exp18b.recover --archive-incomplete",
            recovery="Only completed hash-verified trials are reused; incomplete trials preserved for inspection"))
    print(process.pid, flush=True)


if __name__ == "__main__":
    main()
