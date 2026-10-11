"""Mathematical/implementation tests and minimal real GPU smokes; no full training."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from exp10 import ROOT,RESULTS
from exp10.config import Trial,METHODS,COUNTS,file_hash,save_json
from exp10.audit import code_hashes,protected_snapshot,audit_pairing,audit_fifo,full_audit

def unit_tests(gpus):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpus[0]))
    with (RESULTS/'unit_tests.log').open('w') as log:
        subprocess.run([sys.executable,'-B','-m','pytest','exp10/tests','-q',
            '-o','cache_dir=exp10/runtime/pytest_cache','--basetemp=exp10/runtime/tests',
            f'--junitxml={RESULTS}/tests.xml'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def smoke_jobs(gpus):
    cv=[Trial('cv',method,.003,30,.3 if method==METHODS[0] else 2,stage='smoke',smoke_phase=phase)
        for phase in ('search','final') for method in METHODS]
    # One wave verifies both NLP slots on every permitted physical GPU.
    nlp=[Trial('nlp',method,.005,20,.3 if method==METHODS[0] else 2,seed=20261101+i,stage='smoke')
         for method in METHODS for i in range(len(gpus))]
    return cv,nlp

def verify_platform(gpus):
    from exp10.data import prepare
    from exp10.matrices import prepare_matrices
    from exp10.launcher import run_queue
    from exp10.stage2 import plan
    from exp10.search import initial_jobs,immutable_json
    protocol=plan(gpus);RESULTS.mkdir(parents=True,exist_ok=True)
    before=protected_snapshot();existing=RESULTS/'protected_files_before.json'
    if existing.exists(): assert before==json.loads(existing.read_text())
    else: save_json(existing,before)
    save_json(RESULTS/'platform_validation.json',dict(status='running',full_experiments_started=False))
    try:
        prepare();prepare_matrices()
        immutable_json(RESULTS/'initial_search_configs.json',[dict(j.values(),trial_id=j.id) for j in initial_jobs()])
        unit_tests(gpus)
        out=subprocess.check_output([sys.executable,'-B','-m','exp10.stage2','--gpus',*map(str,gpus),'--plan'],cwd=ROOT,text=True)
        assert json.loads(out)['counts']==COUNTS
        (RESULTS/'stage2_cli_plan.json').write_text(out)
        cv,nlp=smoke_jobs(gpus);rows=run_queue(cv,gpus=gpus)+run_queue(nlp,gpus=gpus)
        assert len(rows)==4+2*len(gpus) and all(r['status']=='completed' for r in rows)
        pairing=audit_pairing(rows);fifo=audit_fifo()
        assert all(fifo['peak_concurrency'][gpu]['nlp']==2 for gpu in gpus)
        assert before==protected_snapshot()
        save_json(RESULTS/'smoke_summary.json',rows);full_audit()
        evidence=('unit_tests.log','tests.xml','smoke_summary.json','assets_manifest.json','stage2_cli_plan.json','initial_search_configs.json')
        save_json(RESULTS/'platform_validation.json',dict(status='passed',full_experiments_started=False,
            code_sha256=code_hashes(),evidence_sha256={n:file_hash(RESULTS/n) for n in evidence},
            smoke_trials=len(rows),smoke_logical_steps_each=3,smoke_horizons=[225,250,310],gpus=gpus,
            pairing=pairing,fifo=fifo,historical_files_unchanged=True,planned_full_trials=COUNTS))
        (RESULTS/'stage1_report.md').write_text(f'# Exp10 Stage 1\n\nUnit tests and {len(rows)} real GPU smokes passed. Each smoke uses three logical steps, including the b=2 boundary. No full training started. GPUs: {gpus}.\n')
        print(f'Exp10 Stage 1 passed; {len(rows)} smokes; complete budget {COUNTS}',flush=True)
    except Exception as error:
        save_json(RESULTS/'platform_validation.json',dict(status='failed',error=str(error),traceback=traceback.format_exc(),full_experiments_started=False))
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',type=int,nargs='+',default=[1,2,3],choices=(1,2,3))
    verify_platform(p.parse_args().gpus)
