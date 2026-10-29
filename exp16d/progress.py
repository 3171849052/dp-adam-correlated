"""Read queue, live workers and completed trial counts."""
import json
from exp16d import RESULTS

def main():
    for name in ('background.json','pipeline_status.json'):
        p=RESULTS/name
        if p.exists():print(name+':\n'+p.read_text())
    for name in ('smoke','search','review','controlled','formal'):
        p=RESULTS/f'{name}_status.json'
        if not p.exists():continue
        rows=json.loads(p.read_text());counts={}
        for r in rows.values():counts[r['status']]=counts.get(r['status'],0)+1
        print(name,counts)
        for r in rows.values():
            if r['status']=='running':print('  GPU',r['gpu'],'PID',r['pid'],r['method'],'log:',r['log'])
if __name__=='__main__':main()
