"""Detach the coordinator into its own session with all stdio redirected."""
import fcntl
import os
import subprocess
import time
from exp18d.common import BASE, ROOT, PYTHON, now, save, verify_seal


def main():
    verify_seal()
    with (BASE / 'runtime/launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pidfile = BASE / 'pipeline.pid'
        if pidfile.exists():
            os.kill(int(pidfile.read_text()),0)
            raise RuntimeError('Existing coordinator; inspect before relaunch')
        command = [PYTHON,'-B','-m','exp18d.pipeline']
        with (BASE / 'logs/pipeline.log').open('a') as log:
            process = subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        pidfile.write_text(str(process.pid)+'\n')
        save(BASE / 'background.json',dict(pid=process.pid,time=now(),command=command,
            detached_session=True,stdin='/dev/null',log='exp18d/logs/pipeline.log'))
        time.sleep(2)
        assert process.poll() is None, 'Coordinator exited; inspect pipeline.log'
        print(f'Exp18d started: PID {process.pid}; independent session; log exp18d/logs/pipeline.log',flush=True)

if __name__ == '__main__': main()
