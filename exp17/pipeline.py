import fcntl
import os
import traceback
from exp17.common import BASE, RESULTS, METHODS, PROTOCOL, save, read, sha, code_hashes, assets
from exp17.search import coarse, fine, rank
from exp17.scheduler import run_phase


def run():
    acceptance=read(RESULTS/'acceptance.json')
    if acceptance['freeze_sha256']!=sha(RESULTS/'protocol_freeze.json'): raise RuntimeError('Acceptance/freeze mismatch')
    frozen=read(RESULTS/'protocol_freeze.json')
    if frozen['code']!=code_hashes() or frozen['assets']!=assets(): raise RuntimeError('Source/assets mismatch')
    c=run_phase('coarse',coarse())
    f=run_phase('fine',fine(c))
    reviews=[]
    for method in METHODS:
        for candidate in rank(c+f,method)[:2]:
            for seed in PROTOCOL['review_seeds']:
                reviews.append(dict(candidate['config'],stage='review',seed=seed))
    r=run_phase('review',reviews)
    selection={method:rank(c+f+r,method) for method in METHODS}
    final_configs={m:selection[m][0]['config'] for m in METHODS}
    freeze=dict(configs=final_configs,selection=selection,protocol_sha256=sha(RESULTS/'protocol_freeze.json'),
                evidence_sha256={name:sha(RESULTS/name) for name in ('coarse_results.json','fine_results.json','review_results.json')})
    path=RESULTS/'final_freeze.json'
    if path.exists() and read(path)!=freeze: raise RuntimeError('Final selection freeze mismatch')
    save(path,freeze)
    trials=[dict(final_configs[m],stage='final',seed=s) for s in PROTOCOL['final_seeds'] for m in METHODS]
    run_phase('final',trials)
    from exp17.report import report
    if not (RESULTS/'ablation_plan.json').exists(): report()
    ablations=[dict(final_configs[m],stage='ablation',seed=s,correction=False) for s in PROTOCOL['final_seeds'] for m in ('lc','hc')]
    run_phase('ablation',ablations)
    report()
    save(RESULTS/'pipeline_status.json',dict(phase='completed',pid=os.getpid(),main_runs=248,ablation_runs=20))

if __name__=='__main__':
    with (BASE/'runtime/scheduler.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run()
        except Exception:
            save(RESULTS/'pipeline_failure.json',dict(pid=os.getpid(),traceback=traceback.format_exc()))
            raise
