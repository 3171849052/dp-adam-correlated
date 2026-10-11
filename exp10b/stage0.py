"""Unit tests and three-step GPU smokes; never starts full training."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import traceback
from exp10b import ROOT,RESULTS
from exp10b.config import Trial,COUNTS,file_hash,save_json,immutable_json
from exp10b.protocol import plan
from exp10b.audit import code_hashes,protected_snapshot,audit_pairing,audit_fifo,full_audit

def unit_tests(gpus):
    with (RESULTS/'unit_tests.log').open('w') as log:
        subprocess.run([sys.executable,'-B','-m','pytest','exp10b/tests','-q',
            '-o','cache_dir=exp10b/runtime/pytest_cache','--basetemp=exp10b/runtime/tests',
            f'--junitxml={RESULTS}/tests.xml'],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpus[0])),
            stdout=log,stderr=subprocess.STDOUT,check=True)

def verify_platform(gpus):
    from exp10b.data import prepare
    from exp10b.matrices import prepare_matrices
    from exp10b.launcher import run_queue
    RESULTS.mkdir(parents=True,exist_ok=True)
    with (RESULTS/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=plan(gpus);before=protected_snapshot()
        immutable_json(RESULTS/'protected_files_before.json',before)
        immutable_json(RESULTS/'full_plan.json',protocol)
        save_json(RESULTS/'platform_validation.json',dict(status='running',full_experiments_started=False))
        try:
            print('Stage 0: prepare local data and 18 horizon/lambda matrices.',flush=True)
            prepare();prepare_matrices()
            print('Stage 0: mathematical, clipping, privacy and workflow tests.',flush=True)
            unit_tests(gpus)
            out=subprocess.check_output([sys.executable,'-B','-m','exp10b.run','--gpus',*map(str,gpus),'--plan'],cwd=ROOT,text=True)
            assert json.loads(out)==protocol
            (RESULTS/'cli_plan.json').write_text(out)
            cv=[Trial('cv',lam,stage='smoke',smoke_phase=phase)
                for phase in ('search','final') for lam in (0.,.1)]
            print('Stage 0: CV smokes for both horizons and lambda zero/positive.',flush=True)
            rows=run_queue(cv,gpus=gpus)
            pilot=[Trial('nlp',0.,20261101+i,'smoke') for i in range(len(gpus))]
            print('Stage 0: single-worker NLP memory pilots.',flush=True)
            pilots=run_queue(pilot,gpus=gpus,nlp_per_gpu=1);rows+=pilots
            # Only enable two NLP workers after measured single-worker feasibility.
            output=subprocess.check_output(['nvidia-smi','--id='+','.join(map(str,gpus)),
                '--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
            available={int(line.split(',')[0]):int(line.split(',')[1])-int(line.split(',')[2])
                       for line in output.strip().splitlines()}
            peaks={r['physical_gpu']:r['peak_reserved_bytes']/2**20 for r in pilots}
            assert set(peaks)==set(gpus)
            assert all(2*peaks[g]+512<available[g] for g in gpus), 'Two NLP workers do not fit measured available GPU memory'
            memory=dict(available_mib=available,single_reserved_peak_mib=peaks,safety_margin_mib=512)
            paired=[Trial('nlp',lam,20261104+i,'smoke') for lam in (0.,.1) for i in range(len(gpus))]
            print('Stage 0: simultaneous two-worker NLP memory smokes on each GPU.',flush=True)
            rows+=run_queue(paired,gpus=gpus)
            assert len(rows)==4+3*len(gpus) and all(r['status']=='completed' for r in rows)
            pairing=audit_pairing(rows);fifo=audit_fifo()
            assert all(fifo['peak_concurrency'][gpu]['nlp']==2 for gpu in gpus)
            assert before==protected_snapshot()
            save_json(RESULTS/'smoke_summary.json',rows);full_audit()
            evidence=('unit_tests.log','tests.xml','smoke_summary.json','assets_manifest.json','cli_plan.json','full_plan.json')
            matrices={str(p.relative_to(RESULTS)):file_hash(p) for p in sorted((RESULTS/'matrix_cache').glob('*'))}
            evidence_hashes={n:file_hash(RESULTS/n) for n in evidence};evidence_hashes.update(matrices)
            save_json(RESULTS/'platform_validation.json',dict(status='passed',full_experiments_started=False,
                code_sha256=code_hashes(),evidence_sha256=evidence_hashes,
                smoke_trials=len(rows),smoke_logical_steps_each=3,smoke_horizons=[225,250,310],gpus=gpus,
                nlp_per_gpu=2,memory=memory,pairing=pairing,fifo=fifo,
                historical_files_unchanged=True,planned_full_trials=COUNTS))
            (RESULTS/'stage0_report.md').write_text(f'# Exp10b Stage 0\n\nAll tests and {len(rows)} three-step real GPU smokes passed. Horizons: 225/250/310. GPUs: {gpus}. Two NLP workers per GPU passed after single-worker memory measurement. No complete training or official evaluation ran. Full budget: 36 + 84 + 40 = 160.\n')
            print(f'Exp10b Stage 0 passed: {len(rows)} smokes; full budget {COUNTS}',flush=True)
        except Exception as error:
            save_json(RESULTS/'platform_validation.json',dict(status='failed',error=str(error),traceback=traceback.format_exc(),full_experiments_started=False))
            raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',type=int,nargs='+',choices=(1,2,3),default=[1,2,3])
    verify_platform(parser.parse_args().gpus)
