"""Launch the accepted formal experiment with nohup and a detached session."""
import os
import subprocess
from exp18g.common import *

def start():
    assert not (BASE/'launch.json').exists()
    assert read(BASE/'acceptance/preflight.json')['passed']
    verify_seal()
    command=['nohup','conda','run','--no-capture-output','-n','curve','python','-B','-m','exp18g.pipeline']
    with (BASE/'logs/pipeline.log').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
    save(BASE/'launch.json',dict(pid=process.pid,command=command,log='exp18g/logs/pipeline.log',time=now()))
    (BASE/'logs/pipeline.pid').write_text(str(process.pid)+'\n')
    print(f'Started background launcher PID {process.pid}',flush=True)
if __name__=='__main__':start()
