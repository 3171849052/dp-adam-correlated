"""Start or safely resume the accepted full experiment in a detached session."""
import fcntl
import os
import subprocess
import time
from exp17.common import BASE, ROOT, RESULTS, PYTHON, save, read, sha


def start():
    accepted=read(RESULTS/'acceptance.json')
    if accepted['freeze_sha256']!=sha(RESULTS/'protocol_freeze.json'): raise RuntimeError('Unaccepted freeze')
    # The child acquires this lock too; the launcher lock serializes concurrent starters.
    with (BASE/'runtime/launcher.lock').open('w') as launcher:
        fcntl.flock(launcher,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (BASE/'runtime/scheduler.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (RESULTS/'pipeline.log').open('a') as log:
            p=subprocess.Popen([PYTHON,'-B','-m','exp17.pipeline'],cwd=ROOT,
                env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)
        time.sleep(2)
        if p.poll() is not None:raise RuntimeError('Pipeline exited; inspect pipeline.log')
        state=dict(pid=p.pid,session_id=os.getsid(p.pid),gpus=[1,2,3],log=str(RESULTS/'pipeline.log'),started_unix=time.time())
        save(RESULTS/'background.json',state);print(state)

if __name__=='__main__':start()
