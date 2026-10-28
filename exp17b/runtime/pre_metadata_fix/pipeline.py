"""Frozen A -> B -> C -> C2 -> D, with validation-only decisions."""
import fcntl
import os
import traceback
import numpy as np
from exp17b.common import BASE, RESULTS, PROTOCOL, LR, R, save, read, sha, code_hashes, assets
from exp17b.scheduler import run_phase


def configurations(stage, choices, seeds, steps):
    return [dict(stage=stage,method=m,alpha=a,r=r,lr=LR,seed=s,steps=steps) for s in seeds for m,a,r in choices]


def select_alpha(results, method):
    candidates=[]
    for alpha in [.25,.5,1.]:
        rows=[x for x in results if x['config']['method']==method and x['config']['alpha']==alpha]
        assert len(rows)==3
        candidates.append((np.mean([x['validation_loss'] for x in rows]),-np.mean([x['accuracy'] for x in rows]),alpha))
    return float(min(candidates)[2])


def improvement(results,r,seeds):
    pairs={}
    for method in ['smooth','gh']:
        pairs[method]={x['config']['seed']:x['validation_loss'] for x in results if x['config']['method']==method and x['config']['r']==r and x['config']['seed'] in seeds}
        assert set(pairs[method])==set(seeds)
    return float(np.mean([pairs['smooth'][s]-pairs['gh'][s] for s in seeds]))


def select_extra(validation,ablation):
    seeds=PROTOCOL['C2']['seeds']
    gains={r:improvement(ablation,r,seeds) for r in [0.,1.]}
    main=improvement(validation,R,seeds)
    eligible=[r for r in gains if gains[r]>0 and gains[r]>main]
    selected=min(eligible,key=lambda r:(-gains[r],r)) if eligible else None
    return dict(selected_r=selected,main_improvement=main,ablation_improvements=gains,seeds=seeds)


def run():
    acceptance=read(RESULTS/'acceptance.json')
    assert acceptance['freeze_sha256']==sha(RESULTS/'protocol_freeze.json')
    frozen=read(RESULTS/'protocol_freeze.json')
    assert frozen['code']==code_hashes() and frozen['assets']==assets()
    from exp17b.report import report
    report()
    save(RESULTS/'pipeline_status.json',dict(phase='A',pid=os.getpid(),status='running'))
    from exp17b.monte_carlo import run as mc
    mc();report()
    choices=[('smooth',0.,R)]+[(m,a,R) for m in ['lc','gh'] for a in [.25,.5,1.]]
    b=run_phase('B',configurations('B',choices,PROTOCOL['B']['seeds'],50))
    chosen={m:select_alpha(b,m) for m in ['lc','gh']}
    save(RESULTS/'selected_alphas.json',chosen);report()
    main=[('smooth',0.,R)]+[(m,chosen[m],R) for m in ['lc','gh']]
    c=run_phase('C',configurations('C',main,PROTOCOL['C']['seeds'],225));report()
    ablation=[(m,0. if m=='smooth' else chosen['gh'],r) for r in [0.,1.] for m in ['smooth','gh']]
    c2=run_phase('C2',configurations('C2',ablation,PROTOCOL['C2']['seeds'],225));report()
    decision=select_extra(c,c2)
    if decision['selected_r'] is not None:
        main += [('gh',chosen['gh'],decision['selected_r']),('smooth',0.,decision['selected_r'])]
    final=configurations('D',main,PROTOCOL['D']['seeds'],250)
    freeze=dict(configs=final,decision=decision,alphas=chosen,protocol_sha256=sha(RESULTS/'protocol_freeze.json'),
                evidence_sha256={n:sha(RESULTS/n) for n in ['B_results.json','C_results.json','C2_results.json']})
    path=RESULTS/'final_freeze.json'
    if path.exists(): assert read(path)==__import__('json').loads(__import__('json').dumps(freeze))
    save(path,freeze)
    run_phase('D',final);report()
    save(RESULTS/'pipeline_status.json',dict(phase='completed',pid=os.getpid(),runs=48+len(final)))

if __name__=='__main__':
    with (BASE/'runtime/scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run()
        except Exception:
            save(RESULTS/'pipeline_failure.json',dict(pid=os.getpid(),traceback=traceback.format_exc()))
            raise
