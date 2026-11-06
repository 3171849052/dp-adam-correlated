"""Detached coordinator survives the initiating shell and Codex session."""
import fcntl
import os
import shlex
import subprocess
import time
from exp18e.common import BASE,ROOT,PYTHON,save,now,verify_seal


def main():
    verify_seal()
    with (BASE/'runtime/launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pidfile=BASE/'logs/pipeline.pid'
        assert not pidfile.exists(),'Existing launch record requires inspection'
        command=[PYTHON,'-B','-m','exp18e.pipeline']
        with (BASE/'logs/pipeline.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        pidfile.write_text(str(process.pid)+'\n')
        (BASE/'logs/launch_command.txt').write_text(shlex.join(command)+'\n')
        save(BASE/'logs/background.json',dict(pid=process.pid,time=now(),command=command,cwd=str(ROOT),
            independent_session=True,stdin='/dev/null',stdout_stderr='exp18e/logs/pipeline.log'))
        time.sleep(3)
        assert process.poll() is None,'Coordinator exited: inspect exp18e/logs/pipeline.log'
        print(f'Exp18e background PID {process.pid}; log exp18e/logs/pipeline.log',flush=True)

if __name__=='__main__':main()
