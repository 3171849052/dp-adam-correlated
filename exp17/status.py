"""Read-only progress view; never changes workers or result files."""
import json
import os
from exp17.common import RESULTS, read

if __name__=='__main__':
    for name in ('background.json','pipeline_status.json','pipeline_failure.json'):
        path=RESULTS/name
        if path.exists():
            value=read(path)
            if 'pid' in value:
                try:os.kill(value['pid'],0);value['process_alive']=True
                except ProcessLookupError:value['process_alive']=False
            print(name,json.dumps(value,indent=2))
    path=RESULTS/'pipeline_status.json'
    if path.exists():
        for item in read(path).get('active',[]):
            progress=RESULTS/'trials'/item['trial']/'progress.json'
            print(item['trial'],json.dumps(read(progress) if progress.exists() else {'state':'initializing'}))
