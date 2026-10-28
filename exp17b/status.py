import os
from exp17b.common import RESULTS, read
for name in ['background.json','pipeline_status.json','mc_progress.json','pipeline_failure.json']:
    path=RESULTS/name
    if path.exists():
        data=read(path)
        if 'pid' in data:data['process_exists']=os.path.exists('/proc/'+str(data['pid']))
        print(name,data)
