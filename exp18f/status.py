"""Read live stage/task progress without importing torch or touching experiment state."""
import json
from exp18f.common import BASE,read,outdir

def status():
    pipeline=read(BASE/'pipeline_status.json')
    states=[read(p) for p in (BASE/'logs').glob('*_status.json')]
    running=[s for s in states if s['status']=='running']
    for state in running:
        for worker in state['active']:
            path=BASE/worker['directory']/'steps.jsonl'
            trace=path.read_text().splitlines() if path.exists() else []
            worker['step']=json.loads(trace[-1])['step'] if trace else 0
    return dict(pipeline=pipeline,stages=states,launch=read(BASE/'launch.json') if (BASE/'launch.json').exists() else None)

if __name__=='__main__':print(json.dumps(status(),indent=2))
