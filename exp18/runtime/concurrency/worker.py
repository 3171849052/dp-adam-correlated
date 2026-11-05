import sys,time,os
from pathlib import Path
from exp18.common import read,save
from exp18.optimizer import Update
from exp18.train import run
p=Path(sys.argv[1]);batch=p.parent.parent;n=int(sys.argv[2]);old=Update.step

def step(self,q):
    if self.t==0:
        save(p.parent/'ready.json',dict(pid=os.getpid(),time=time.time()))
        deadline=time.time()+180
        while len(list(batch.glob('*/ready.json')))<n:
            if (batch/'abort').exists():raise RuntimeError('Peer failure in concurrency benchmark')
            if time.time()>deadline:raise TimeoutError('CUDA concurrency barrier')
            time.sleep(.1)
    return old(self,q)
Update.step=step
try:run(p)
except Exception:
    import traceback
    (p.parent/'error.txt').write_text(traceback.format_exc());raise
