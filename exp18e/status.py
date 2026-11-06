"""Read-only runtime snapshot (also usable while search has not yet frozen)."""
import os
from exp18e.common import BASE,read,save,now


def status():
    state=read(BASE/'logs/pipeline_status.json')
    pid=int((BASE/'logs/pipeline.pid').read_text())
    os.kill(pid,0)
    stages={p.stem:read(p) for p in (BASE/'logs').glob('*_status.json') if p.name!='pipeline_status.json'}
    result=dict(time=now(),coordinator_alive=True,pid=pid,pipeline=state,stages=stages)
    if (BASE/'search/counts.json').exists():result['search']=read(BASE/'search/counts.json')
    print(__import__('json').dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':status()
