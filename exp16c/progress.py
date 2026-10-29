"""Read-only progress view, including logical steps of active trials."""
from collections import Counter
from exp16c import RESULTS
from exp16c.support import read
import json

def main():
    result={}
    for name in ('background','pipeline_status'):
        p=RESULTS/f'{name}.json'
        if p.exists():result[name]=read(p)
    for stage in ('smoke','search','review','formal'):
        p=RESULTS/f'{stage}_status.json'
        if not p.exists():continue
        states=read(p);active=[]
        for trial_id,r in states.items():
            if r['status']!='running':continue
            path=RESULTS/r['stage']/'cv'/trial_id/'diagnostics.jsonl'
            lines=path.read_text().splitlines() if path.exists() else []
            active.append(dict(trial_id=trial_id,method=r['method'],seed=r['seed'],gpu=r['gpu'],pid=r['pid'],logical_steps=len(lines),log=r['log']))
        result[stage]=dict(counts=dict(Counter(r['status'] for r in states.values())),active=active)
    save=RESULTS/'progress.json'
    from exp16c.config import save_json
    save_json(save,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
