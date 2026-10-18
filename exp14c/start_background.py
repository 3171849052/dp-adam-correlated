import json
import os
import shlex
import subprocess
import sys
from exp14c import ROOT, RESULTS
from exp14c.config import save_json
from exp14c.support import smoke_jobs, code_hashes, protected_snapshot, reference_dir
from exp14c.audit import audit_row

def main():
    assert json.loads((RESULTS/'tests.json').read_text())['status']=='passed'
    assert json.loads((RESULTS/'source_provenance.json').read_text())==code_hashes()
    checks=[]
    from dataclasses import replace
    for cfg in smoke_jobs():
        row=audit_row(json.loads((cfg.output/'summary.json').read_text()))
        ref=json.loads((reference_dir(replace(cfg,stage='final'))/'summary.json').read_text())
        for k in ('initialization_sha256','classifier_initialization_sha256'): assert row[k]==ref[k]
        checks.append(dict(trial_id=cfg.id,task=cfg.task,method=cfg.method,horizon=cfg.total_steps,gpu=row['physical_gpu'],steps=2,finite=True,initialization_exact=True))
    save_json(RESULTS/'smoke_checks.json',checks)
    assert protected_snapshot()==json.loads((RESULTS/'protected_before.json').read_text())
    save_json(RESULTS/'prelaunch_audit.json',dict(status='passed',history_unchanged=True,smokes=len(checks),source_verified=True))
    assert not (RESULTS/'background.json').exists()
    command=[sys.executable,'-B','-m','exp14c.pipeline']
    with (RESULTS/'scheduler.log').open('w') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record=dict(pid=p.pid,command=shlex.join(command),launch_command='conda run --no-capture-output -n curve python -B -m exp14c.start_background',cwd=str(ROOT),start_new_session=True,log=str(RESULTS/'scheduler.log'))
    save_json(RESULTS/'background.json',record);print(json.dumps(record,indent=2))
if __name__=='__main__': main()
