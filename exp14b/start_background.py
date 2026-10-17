import json
import os
import shlex
import subprocess
import sys
from exp14b import ROOT, RESULTS
from exp14b.config import save_json
from exp14b.support import jobs, reference_dir, code_hashes

def main():
    assert json.loads((RESULTS/'tests.json').read_text())['status']=='passed'
    assert json.loads((RESULTS/'source_provenance.json').read_text())==code_hashes()
    checks=[]
    for cfg in jobs('smoke'):
        row=json.loads((cfg.output/'summary.json').read_text())
        assert row['status']=='completed' and row['finite'] and row['optimizer_steps']==2
        assert not row['official_evaluation_used']
        diagnostics=[json.loads(s) for s in (cfg.output/'optimizer_metrics.jsonl').read_text().splitlines()]
        assert len(diagnostics)==2 and all(s['nonfinite_parameter_tensors']==0 for s in diagnostics)
        # Initialization is independent of split and exactly paired with the formal seed.
        from dataclasses import replace
        ref=json.loads((reference_dir(replace(cfg,stage='final'))/'summary.json').read_text())
        for key in ('initialization_sha256','classifier_initialization_sha256'):
            assert row[key]==ref[key],key
        checks.append(dict(method=cfg.method,gpu=row['physical_gpu'],steps=2,finite=True,initialization_exact=True))
    save_json(RESULTS/'smoke_checks.json',checks)
    assert not (RESULTS/'background.json').exists()
    command=[sys.executable,'-B','-m','exp14b.pipeline']
    with (RESULTS/'scheduler.log').open('w') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record=dict(pid=p.pid,command=shlex.join(command),launch_command='conda run --no-capture-output -n curve python -B -m exp14b.start_background',cwd=str(ROOT),start_new_session=True,log=str(RESULTS/'scheduler.log'))
    save_json(RESULTS/'background.json',record)
    print(json.dumps(record,indent=2))
if __name__=='__main__': main()
