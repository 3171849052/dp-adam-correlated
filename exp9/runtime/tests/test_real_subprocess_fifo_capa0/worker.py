import json,os,sys,time
from pathlib import Path
j=json.loads(sys.argv[1]);d=Path(sys.argv[2]);time.sleep(.25)
j.update(status='completed',trial_id=sys.argv[3],gpu=os.environ['CUDA_VISIBLE_DEVICES'])
(d/'summary.json').write_text(json.dumps(j))
