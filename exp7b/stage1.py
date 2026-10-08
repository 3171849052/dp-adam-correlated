"""Unit tests and seven one-step GPU smokes; never starts full experiments."""
import argparse
import subprocess
import sys
from exp7b import BASE, ROOT
from exp7b.config import *
from exp7b.launcher import run_queue
from exp7b.frozen import import_fixed
from exp7b.audit import audit_trial, audit_pairing, audit_fifo, array_hash

def smoke_jobs():
    fixed = import_fixed()
    settings = {SGD:(.002,10,None), MOMENTUM_SCALE:(.005,100,.1),
                BIAS:(.005,30,None), BIAS_SCALE:(.005,100,.1)}
    return [trial(m,*(tuple(fixed[m][k] for k in ('lr','C','eps_scale')) if m in fixed else settings[m])) for m in METHODS]

def verify_platform(gpus):
    assert sorted(gpus) == [1,2,3], 'Use physical GPUs 1, 2, 3'
    with (BASE / 'results/unit_tests.log').open('w') as log:
        code = subprocess.call([sys.executable,'-B','-m','pytest','exp7b/tests','-q',
                                '-o','cache_dir=exp7b/runtime/pytest_cache',
                                '--basetemp=exp7b/runtime/tests'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert code == 0, 'See exp7b/results/unit_tests.log'
    results = run_queue(smoke_jobs(),gpus,category='smoke')
    assert len(results) == 7
    for r in results:
        audit_trial(ROOT / r['result_dir'])
    audit_pairing(results)
    save_json(BASE / 'results/smoke_summary.json', results)
    fifo = audit_fifo()
    from exp7b.bandinvmf import bias_matrices, workload_error
    d, strategy, W, metadata = bias_matrices()
    save_json(BASE / 'results/bias_matrix_audit.json',dict(metadata,coefficients=d.tolist(),
              workload_sha256=array_hash(W),strategy_sha256=array_hash(strategy),
              workload_error=workload_error(W,d)))
    save_json(BASE / 'results/platform_validation.json',dict(status='passed',unit_tests='passed',
              smoke_trials=7,smoke_methods=list(METHODS),gpus=gpus,pairing='passed',fifo=fifo,
              protocol_steps=250,protocol_epochs=5,smoke_steps=1,smoke_epochs=1,
              smoke_excluded_from_search=True,full_experiments_started=False))
    print('Stage 1 passed: unit tests, seven GPU smokes, matrices/privacy/hashes, pairing and FIFO.',flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',type=int,nargs=3,default=[1,2,3])
    verify_platform(parser.parse_args().gpus)
