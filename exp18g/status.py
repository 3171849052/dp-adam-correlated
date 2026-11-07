"""Read-only progress; no task restart."""
import json
from exp18g.common import *

def status():
    rows=[]
    for c in configs():
        d=outdir(c);p=d/'steps.jsonl';lines=p.read_text().splitlines() if p.exists() else []
        rows.append(dict(seed=c['seed'],steps=len(lines),complete=(d/'complete.json').exists(),
            process=read(d/'process.json') if (d/'process.json').exists() else None))
    print(json.dumps(dict(pipeline=read(BASE/'pipeline_status.json'),runs=rows),indent=2))
if __name__=='__main__':status()
