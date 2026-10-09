"""Exactly 70 paired five-epoch jobs, using frozen hyperparameters."""
import argparse
from exp8b.config import *
from exp8b.frozen import load_frozen
from exp8b.launcher import run_queue
from exp8b.audit import check_platform,audit_pairing


def final_jobs():
    settings=load_frozen()
    return [trial(m,settings[m]['lr'],settings[m]['C'],settings[m]['eps_scale'],seed) for seed in FINAL_SEEDS for m in METHODS]


def final(gpu=0,queue=None):
    check_platform()
    runner=run_queue if queue is None else queue.run
    jobs=final_jobs()
    # The first download must finish before parallel workers can read the file.
    from exp8b.data import official_validation
    official_validation('final')
    results=runner(jobs,gpu=gpu,category='final')
    assert len(results)==70 and all(r['status']=='completed' for r in results),'Failed final trials are recorded and require inspection'
    audit_pairing(results)
    from exp8b.report import final_report
    final_report()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);final(p.parse_args().gpu)
