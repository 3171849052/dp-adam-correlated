import json
import os
import shlex
import subprocess
import sys
import time
from exp14d import ROOT, RESULTS
from exp14d.config import save_json, file_hash
from exp14d.support import smoke_jobs, code_hashes, protected_snapshot, reference_dir

def main():
    assert sys.prefix.endswith('/envs/curve'),sys.prefix
    assert json.loads((RESULTS/'tests.json').read_text())['status']=='passed'
    assert json.loads((RESULTS/'source_provenance.json').read_text())==code_hashes()
    checks=[]
    for c in smoke_jobs():
        r=json.loads((c.output/'summary.json').read_text());ref=json.loads((reference_dir(c)/'summary.json').read_text())
        assert r['status']=='completed' and r['finite'] and r['optimizer_steps']==r['noise_draws']==2
        assert not r['official_evaluation_used'] and r['physical_batch_size']==c.physical_batch
        for k in ('initialization_sha256','classifier_initialization_sha256'):
            # Search smoke deliberately uses a formal seed; compare against same-seed formal initialization.
            from dataclasses import replace
            formal_ref=json.loads((reference_dir(replace(c,smoke_phase='final'))/'summary.json').read_text())
            assert r[k]==formal_ref[k]
        assert abs(r['calibration']['verified_full_epsilon']-8)<1e-7
        for name,h in r['file_sha256'].items():assert file_hash(c.output/name)==h
        checks.append(dict(trial_id=c.id,task=c.task,method=c.method,horizon=c.total_steps,gpu=r['physical_gpu'],steps=2,finite=True,initialization_exact=True))
    assert {c['gpu'] for c in checks}=={0,1,2,3}
    save_json(RESULTS/'smoke_checks.json',checks)
    assert protected_snapshot()==json.loads((RESULTS/'protected_before.json').read_text())
    save_json(RESULTS/'prelaunch_audit.json',dict(status='passed',history_unchanged=True,smokes=len(checks),source_verified=True))
    assert not (RESULTS/'background.json').exists()
    command=[sys.executable,'-B','-m','exp14d.pipeline']
    with (RESULTS/'scheduler.log').open('w') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record=dict(pid=p.pid,command=shlex.join(command),launch_command='conda run --no-capture-output -n curve python -B -m exp14d.start_background',cwd=str(ROOT),start_new_session=True,log=str(RESULTS/'scheduler.log'),launched_at=time.time())
    save_json(RESULTS/'background.json',record);print(json.dumps(record,indent=2))
if __name__=='__main__':main()
