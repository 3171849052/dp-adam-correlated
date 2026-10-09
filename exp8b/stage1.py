"""Prepare offline assets, verify correctness, then run seven real GPU smokes."""
import argparse
import json
import subprocess
import sys
import traceback
from exp8b import ROOT,RESULTS
from exp8b.config import *
from exp8b.data import download,prepare
from exp8b.launcher import run_queue
from exp8b.audit import audit_pairing,source_hashes,audit_previous_experiments,audit_fifo


def unit_tests(prefix=""):
    log_path=RESULTS/f"{prefix}unit_tests.log"
    xml_path=RESULTS/f"{prefix}tests.xml"
    with log_path.open('w') as log:
        code=subprocess.call([sys.executable,'-B','-m','pytest','exp8b/tests','-q',
             '-o','cache_dir=exp8b/runtime/pytest_cache','--basetemp=exp8b/runtime/tests',
             f'--junitxml={xml_path}'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert code==0,'Tests failed: see exp8b/results/unit_tests.log'


def verify_platform(gpu=0):
    RESULTS.mkdir(parents=True,exist_ok=True)
    save_json(RESULTS/'platform_validation.json',dict(status='running',full_experiments_started=False))
    try:
        download();prepare();unit_tests()
        jobs=[trial(m,1e-4,1.,.1 if METHODS[m]['geometry']=='scale' else None) for m in METHODS]
        results=run_queue(jobs,gpu=gpu,category='smoke')
        assert len(results)==7 and all(r['status']=='completed' for r in results)
        assert all('3080 Ti' in r['device'] for r in results)
        audit_pairing(results)
        save_json(RESULTS/'smoke_summary.json',results)
        from exp8b.report import write_csv
        keys=('method','physical_batch_size','max_length','peak_allocated_bytes','peak_reserved_bytes','logical_step_seconds','samples_per_second')
        write_csv(RESULTS/'smoke_summary.csv',[{k:r[k] for k in keys} for r in results])
        previous=audit_previous_experiments()
        save_json(RESULTS/'platform_validation.json',dict(status='passed',unit_tests='passed',unit_tests_sha256=file_hash(RESULTS/'unit_tests.log'),
                  tests_xml_sha256=file_hash(RESULTS/'tests.xml'),smoke_methods=list(METHODS),smoke_steps_each=1,gpu=gpu,
                  source_hashes=source_hashes(),smoke_summary_hashes={r['method']:file_hash(ROOT/r['result_dir']/'summary.json') for r in results},
                  pairing='passed',previous_experiments=previous,physical_batch_size=1000,max_length=128,
                  fifo=audit_fifo(),
                  protocol_steps=310,protocol_epochs=5,epsilon=8,delta=1e-5,
                  official_validation_used=False,full_experiments_started=False))
        lines=['# Exp8b Stage 1','','Unit tests and seven full logical-step GPU smokes passed. Stage 2 has not started.',
               'GPU: RTX 3080 Ti; physical=logical batch 1000; max_length 128; FP32; two backwards, calibrated DP noise, ordinary Adam.',
               'One-step smoke is a feasibility check; it is not evidence of five-epoch convergence or accuracy.',
               'Cold step timings include kernel warm-up. Peak memory includes persistent model, Adam and noise buffers.',
               '', '| Method | Allocated GiB | Reserved GiB | Seconds / step | Samples / s |','|---|---:|---:|---:|---:|']
        lines += [f"| {r['method']} | {r['peak_allocated_bytes']/2**30:.3f} | {r['peak_reserved_bytes']/2**30:.3f} | {r['logical_step_seconds']:.3f} | {r['samples_per_second']:.1f} |" for r in results]
        lines += ['','All strategies rebuilt for T=310; GDP calibrated from their actual fixed-participation sensitivity.',
                  'Repeated tokens, padding, position embeddings, token-type embeddings, arbitrary coordinate scales, clipping coefficients and clipped gradient sums checked against explicit per-example autograd.',
                  '', 'From repository root:', '', '```bash','conda run --no-capture-output -n curve python -B -m exp8b.stage2 --gpu 0','```']
        (RESULTS/'stage1_report.md').write_text('\n'.join(lines)+'\n')
        print('\n'.join(lines),flush=True)
    except Exception as error:
        save_json(RESULTS/'platform_validation.json',dict(status='failed',error=str(error),physical_batch_size=1000,
                  traceback=traceback.format_exc(),full_experiments_started=False,
                  policy='Stop; never reduce the physical batch or launch Stage 2 after OOM'))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);verify_platform(p.parse_args().gpu)
