"""Start one durable phase-two scheduler, retaining PID and exact invocation."""
import json
import os
import shlex
import subprocess
import sys
import time
from exp15c import ROOT, RESULTS
from exp15c.config import save_json
from exp15c.search import require_phase1


def main():
    require_phase1()
    state=RESULTS/'background.json'
    if state.exists():
        previous=json.loads(state.read_text())
        try:
            os.kill(previous['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Recorded scheduler is still running')
    argv=[sys.executable,'-B','-m','exp15c.pipeline']
    with (RESULTS/'pipeline.log').open('a') as log:
        p=subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES="1,2,3",PYTHONDONTWRITEBYTECODE="1"),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row=dict(pid=p.pid,started=time.time(),command=shlex.join(argv),log=str(RESULTS/'pipeline.log'))
    save_json(state,row);print(json.dumps(row),flush=True)


if __name__=='__main__':main()
