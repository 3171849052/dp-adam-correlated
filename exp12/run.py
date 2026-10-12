"""Validate, smoke, then automatically execute exactly 20+4+10 CV trainings."""
import argparse
from dataclasses import replace
from itertools import product
import json
import os
import platform
import subprocess
import time
import traceback
from exp12 import BASE, ROOT, RESULTS
from exp12.config import Trial, METHOD, RECHECK_SEEDS, FINAL_SEEDS, file_hash, save_json
from exp12.audit import code_hashes, protected_snapshot, audit_trial, baseline_audit, scheduler_audit
from exp12.launcher import run_queue
from exp12.report import write_csv, generate


def grid_jobs():
    return [Trial('cv',METHOD,lr,C) for lr,C in product((.001,.003,.005,.007),(3,10,30,100,200))]


def rank(rows):
    assert all(r['status']=='completed' for r in rows)
    return sorted(rows,key=lambda r:(-r['accuracy'],r['trial_id']))


def select(top,grid,rechecks):
    assert len(rechecks)==4 and all(r['status']=='completed' for r in rechecks)
    index={r['trial_id']:r for r in grid+rechecks};evidence=[]
    for cfg in top:
        runs=[index[cfg.id]]+[index[replace(cfg,stage='recheck',seed=s).id] for s in RECHECK_SEEDS]
        evidence.append(dict(lr=cfg.lr,C=cfg.C,eps_scale=None,
                       mean_validation_accuracy=sum(r['accuracy'] for r in runs)/3,
                       trial_ids=[r['trial_id'] for r in runs],base_id=cfg.id))
    evidence.sort(key=lambda r:(-r['mean_validation_accuracy'],r['base_id']))
    return evidence[0],evidence


def execute(gpus):
    assert len(gpus)==3 and len(set(gpus))==3 and set(gpus)<=set(range(4))
    assert not (RESULTS/'run_state.json').exists(), 'Existing execution must be inspected; no automatic retry'
    started=time.time()
    save_json(RESULTS/'run_state.json',dict(status='running',started_unix=started,gpus=gpus,pid=os.getpid(),planned_trainings=34))
    save_json(RESULTS/'protected_files_before.json',protected_snapshot())
    from exp12.data import prepare,prepare_official
    from exp12.bandinvmf import bias_matrices
    from exp12.frozen import freeze
    prepare()
    save_json(RESULTS/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
              conda_environment=os.environ.get('CONDA_DEFAULT_ENV'),gpus=gpus,
              nvidia_smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=True).stdout))
    assert os.environ.get('CONDA_DEFAULT_ENV')=='curve'
    for n in (225,250):
        d,S,W,meta=bias_matrices(n)
        print(f'T={n}, old d={meta["old_filter"]["coefficients"]}, new d={d.tolist()}',flush=True)
    # Historical accuracy is not used in selection; this verifies its eligibility as a baseline.
    try:
        baseline_audit()
    except AssertionError as error:
        save_json(RESULTS/'baseline_audit.json',dict(status='ineligible',reason=str(error),traceback=traceback.format_exc()))
        print(f'Historical baseline is ineligible for pairing: {error}',flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpus[0]))
    command=['conda','run','--no-capture-output','-n','curve','python','-B','-m','pytest',
             'exp12/tests','-q','-p','no:cacheprovider','--basetemp='+str(BASE/'runtime/pytest_tmp'),
             '--junitxml='+str(RESULTS/'tests.xml')]
    with (RESULTS/'tests.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Mathematical, clipping, RNG, GDP and protocol tests passed',flush=True)
    # Both planned horizons and all three physical devices are exercised, without evaluation.
    smokes=[Trial('cv',METHOD,.003,30,stage='smoke',smoke_phase='final' if i==1 else 'search',seed=20261101+i) for i in range(3)]
    smoke_rows=run_queue(smokes,gpus=gpus)
    save_json(RESULTS/'smoke_results.json',smoke_rows)
    save_json(RESULTS/'stage1.json',dict(status='passed',code_sha256=code_hashes(),
              tests_sha256=file_hash(RESULTS/'tests.xml'),smoke_ids=[j.id for j in smokes],
              official_evaluation_used=False,seconds=time.time()-started))
    print('All tests and three-GPU smoke passed; starting full 20+4+10 workflow now',flush=True)
    jobs=grid_jobs()
    save_json(RESULTS/'grid_plan.json',[dict(j.values(),trial_id=j.id) for j in jobs])
    rows=run_queue(jobs,gpus=gpus);assert len(rows)==20
    save_json(RESULTS/'grid_results.json',rows);write_csv(RESULTS/'grid_results.csv',rows)
    top=[Trial(**{k:r[k] for k in Trial.__dataclass_fields__}) for r in rank(rows)[:2]]
    save_json(RESULTS/'top2.json',[j.values() for j in top])
    rechecks=run_queue([replace(j,stage='recheck',seed=s) for j in top for s in RECHECK_SEEDS],gpus=gpus)
    save_json(RESULTS/'recheck_results.json',rechecks);write_csv(RESULTS/'recheck_results.csv',rechecks)
    winner,evidence=select(top,rows,rechecks)
    save_json(RESULTS/'selection_evidence.json',evidence)
    freeze(winner)
    print(f'Frozen winner: LR={winner["lr"]}, C={winner["C"]}',flush=True)
    prepare_official()
    finals=run_queue([Trial('cv',METHOD,winner['lr'],winner['C'],seed=s,stage='final') for s in FINAL_SEEDS],gpus=gpus)
    save_json(RESULTS/'final_results.json',finals);write_csv(RESULTS/'final_results.csv',finals)
    assert len(rows)+len(rechecks)+len(finals)==34
    for r in rows+rechecks+finals:
        audit_trial(Trial(**{k:r[k] for k in Trial.__dataclass_fields__}))
    unchanged=protected_snapshot()==json.loads((RESULTS/'protected_files_before.json').read_text())
    assert unchanged, 'Protected historical/data/cache files changed'
    save_json(RESULTS/'completion.json',dict(status='completed',search=20,recheck=4,final=10,total=34,
              epochs_per_training=5,started_unix=started,finished_unix=time.time(),seconds=time.time()-started,
              protected_files_unchanged=unchanged,code_sha256=code_hashes(),scheduler=scheduler_audit()))
    save_json(RESULTS/'run_state.json',dict(status='completed',gpus=gpus,pid=os.getpid(),started_unix=started,
              seconds=time.time()-started,completed_trainings=34))
    save_json(RESULTS/'progress.json',dict(status='completed',completed_trainings=34,planned_trainings=34))
    generate()
    paths=[p for p in sorted(RESULTS.rglob('*')) if p.is_file() and p.name not in ('result_manifest.json','workflow.log')]
    save_json(RESULTS/'result_manifest.json',dict(code_sha256=code_hashes(),
              files_sha256={str(p.relative_to(RESULTS)):file_hash(p) for p in paths}))
    print('Completed 34/34 full trainings; final report: exp12/results/report.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2]);args=p.parse_args()
    try:
        execute(args.gpus)
    except Exception as error:
        save_json(RESULTS/'failure.json',dict(status='failed',error=f'{type(error).__name__}: {error}',
                  traceback=traceback.format_exc(),time_unix=time.time()))
        save_json(RESULTS/'progress.json',dict(status='failed',error=f'{type(error).__name__}: {error}'))
        generate()
        raise
