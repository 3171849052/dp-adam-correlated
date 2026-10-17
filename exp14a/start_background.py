"""Launch only after all correctness and full-batch GPU smoke checks pass."""
import json
import os
import shlex
import subprocess
import sys
from exp14a import ROOT, RESULTS
from exp14a.config import save_json
from exp14a.support import jobs, reference_dir

def main():
    assert json.loads((RESULTS/'tests.json').read_text())['status']=='passed'
    smoke=json.loads((RESULTS/'smoke_status.json').read_text())
    assert len(smoke)==8 and all(s['status']=='completed' for s in smoke.values())
    checks=[]
    for cfg in jobs('smoke'):
        s=json.loads((cfg.output/'summary.json').read_text())
        ref=json.loads((reference_dir(cfg)/'summary.json').read_text())
        assert s['finite'] and s['optimizer_steps']==2 and s['noise_draws']==2
        for key in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','strategy_sha256','workload_sha256'):
            assert s[key]==ref[key],key
        diagnostics=[json.loads(line) for line in (cfg.output/'optimizer_metrics.jsonl').read_text().splitlines()]
        assert len(diagnostics)==2 and all(r['nonfinite_parameter_tensors']==0 for r in diagnostics)
        checks.append(dict(task=cfg.task,method=cfg.method,gpu=s['physical_gpu'],finite=True,paired_artifacts=True,steps=2))
    save_json(RESULTS/'smoke_checks.json',checks)
    assert not (RESULTS/'background.json').exists(), 'Already launched'
    command=[sys.executable,'-B','-m','exp14a.launch']
    with (RESULTS/'scheduler.log').open('w') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record=dict(pid=p.pid,command=shlex.join(command),launch_command='conda run --no-capture-output -n curve python -B -m exp14a.start_background',cwd=str(ROOT),start_new_session=True,log=str(RESULTS/'scheduler.log'))
    save_json(RESULTS/'background.json',record)
    print(json.dumps(record,indent=2))

if __name__=='__main__': main()
