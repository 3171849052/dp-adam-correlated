"""Start a detached, offline conda controller; persist PID and exact command."""
import os
import subprocess
from exp18f.common import BASE,ROOT,read,save,now,verify_seal

def main():
    verify_seal()
    assert not (BASE/'launch.json').exists(),'A launch is already recorded; inspect it before any restart'
    command=['conda','run','--no-capture-output','-n','curve','python','-B','-m','exp18f.pipeline']
    with (BASE/'logs/pipeline.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,env=os.environ.copy())
    launch=dict(pid=process.pid,process_group=process.pid,command=command,cwd=str(ROOT),time=now(),
        log=str(BASE/'logs/pipeline.log'),detached=True,mechanism='setsid/start_new_session with redirected stdin/stdout/stderr')
    save(BASE/'launch.json',launch);(BASE/'logs/launcher.pid').write_text(str(process.pid)+'\n')
    print(launch,flush=True)

if __name__=='__main__':main()
