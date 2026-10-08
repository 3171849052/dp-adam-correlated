import json,os,sys,time
from pathlib import Path
j=json.loads(sys.argv[1]);d=Path(sys.argv[2]);d.mkdir(parents=True)
time.sleep(.1)
j.update(status='completed',smoke=True,final_test_top1=.1,gpu=os.environ['CUDA_VISIBLE_DEVICES'])
(d/'summary.json').write_text(json.dumps(j))
