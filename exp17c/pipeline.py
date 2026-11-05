"""Validation-only selection followed by a frozen formal comparison."""
import fcntl
import os
import traceback
import numpy as np
from exp17c.common import BASE,RESULTS,PROTOCOL,LR,save,read,sha,code_hashes,assets
from exp17c.scheduler import run_phase
from exp17c.report import report

def configurations(stage,alphas):
    p=PROTOCOL['B' if stage=='B_ext' else stage]
    return [dict(stage=stage,method='lc',alpha=a,r=0.,lr=LR,seed=s,steps=p['steps']) for s in p['seeds'] for a in alphas]

def ranking(rows):
    alphas=sorted({r['config']['alpha'] for r in rows})
    return sorted(alphas,key=lambda a:(np.mean([r['validation_loss'] for r in rows if r['config']['alpha']==a]),
        -np.mean([r['accuracy'] for r in rows if r['config']['alpha']==a]),a))

def run():
    assert read(RESULTS/'acceptance.json')['freeze_sha256']==sha(RESULTS/'protocol_freeze.json')
    f=read(RESULTS/'protocol_freeze.json')
    assert f['code']==code_hashes() and f['assets']==assets()
    report()
    b=run_phase('B',configurations('B',PROTOCOL['alphas']));report()
    order=ranking(b);best=next(a for a in order if a>0)
    boundary=best in [1e-14,1e-10]
    extend=boundary and order.index(best)<order.index(0)
    extra=([1e-15,3e-15] if best==1e-14 else [3e-10,1e-9]) if extend else []
    save(RESULTS/'boundary_check.json',dict(best=best,boundary=boundary,extension=extra,
        reason='Boundary winner beats baseline: test adjacent decade' if extend else 'No boundary winner outperforming baseline'))
    if extra: b+=run_phase('B_ext',configurations('B_ext',extra))
    choices=[0.]+[a for a in ranking(b) if a>0][:3]
    save(RESULTS/'selected_alphas.json',dict(C=choices))
    c=run_phase('C',configurations('C',choices));report()
    best=next(a for a in ranking(c) if a>0)
    final=configurations('D',[0.,best])
    save(RESULTS/'final_freeze.json',dict(configs=final,alpha=best,
        protocol_sha256=sha(RESULTS/'protocol_freeze.json'),C_sha256=sha(RESULTS/'C_results.json'),
        boundary_unresolved=best in ([min(extra),max(extra)] if extra else [1e-14,1e-10])))
    run_phase('D',final);report()
    save(RESULTS/'pipeline_status.json',dict(phase='completed',pid=os.getpid()))

if __name__=='__main__':
    with (BASE/'runtime/scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run()
        except Exception:
            save(RESULTS/'pipeline_failure.json',dict(pid=os.getpid(),traceback=traceback.format_exc()))
            raise
