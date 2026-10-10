"""CPU-only orchestration and report validation with explicitly synthetic fixtures."""
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace
import sys,json,contextlib,io
import numpy as np
from exp9 import BASE,RESULTS
from exp9.config import save_json,Trial,METHODS,COUNTS
from exp9.grid import all_grid
from exp9.search import top_two,recheck_jobs,choose_winners
from exp9.final import formal_jobs
from exp9 import stage2,report

root=BASE/'runtime/workflow_validation';root.mkdir(parents=True,exist_ok=True)
rows=[dict(j.values(),trial_id=j.id,status='completed',accuracy=.65+(i%13)/1000,seconds=1.,train_seconds=.5,
           epochs=[dict(train_loss=.5,clip_fraction=.5)],peak_allocated_bytes=1024) for i,j in enumerate(all_grid())]
top2=top_two(rows)
rechecks=[dict(j.values(),trial_id=j.id,status='completed',accuracy=.66,seconds=1.,train_seconds=.5,
              epochs=[dict(train_loss=.5,clip_fraction=.5)],peak_allocated_bytes=1024) for j in recheck_jobs(top2)]
winners,evidence=choose_winners(top2,rows,rechecks)

checks=[]
for fail in (None,'gate','search','freeze','final','sweep'):
    calls=[]
    output=root/f'pipeline_{fail}';output.mkdir(exist_ok=True)
    def call(name,ret=None):
        def wrapped(*args,**kwargs):
            calls.append(name)
            if name==fail: raise RuntimeError('injected dependency failure')
            if name in ('search','final','sweep'): assert kwargs==dict(gpus=[0,1,2,3],cv_per_gpu=1,nlp_per_gpu=2)
            return ret
        return wrapped
    def formal(stage,**kwargs):return call(stage)(**kwargs)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(stage2,'RESULTS',output))
        stack.enter_context(patch('exp9.audit.verify_stage1',call('gate')))
        stack.enter_context(patch('exp9.stage1.unit_tests',call('tests')))
        stack.enter_context(patch('exp9.grid.write_grid',call('grid')))
        stack.enter_context(patch('exp9.search.search',call('search',winners)))
        stack.enter_context(patch('exp9.frozen.freeze',call('freeze')))
        stack.enter_context(patch('exp9.data.prepare_official',call('official')))
        stack.enter_context(patch('exp9.final.run_formal',formal))
        stack.enter_context(patch('exp9.report.generate_report',call('report')))
        stack.enter_context(patch.object(sys,'argv',['exp9.stage2','--gpus','0','1','2','3','--cv-per-gpu','1','--nlp-per-gpu','2']))
        try:
            with contextlib.redirect_stdout(io.StringIO()):stage2.main()
            assert fail is None
        except RuntimeError:
            assert fail is not None and calls[-1]==fail
    if fail is None: assert calls==['gate','tests','grid','search','freeze','official','final','sweep','report']
    checks.append(dict(injected_failure=fail,calls=calls,status='passed'))

out=root/'synthetic_report';out.mkdir(exist_ok=True)
def synthetic_formal(stage):
    return [dict(j.values(),trial_id=j.id,status='completed',accuracy=.65+.01*METHODS.index(j.method)+(j.seed%7)*.001,
                 seconds=1.,train_seconds=.5,epochs=[dict(train_loss=.5,clip_fraction=.5)],peak_allocated_bytes=1024)
            for j in formal_jobs(winners,stage)]
for name,value in [('grid_results.json',rows),('recheck_results.json',rechecks),('final_results.json',synthetic_formal('final')),
                   ('sweep_results.json',synthetic_formal('sweep')),('selection_evidence.json',evidence)]:save_json(out/name,value)
with patch.object(report,'RESULTS',out),patch('exp9.frozen.load_frozen',return_value=winners),patch('exp9.audit.full_audit',return_value={'synthetic_test':True}):
    report.generate_report()
assert len(json.loads((out/'method_summary.json').read_text()))==14
assert len(json.loads((out/'paired_effects.json').read_text()))==42
assert len(json.loads((out/'privacy_utility.json').read_text()))==56
assert len(list((out/'figures').glob('grid_*.png')))==14
assert 'fixed-hyperparameter' in (out/'final_report.md').read_text()
save_json(RESULTS/'workflow_validation.json',dict(status='passed',mocked_orchestration_no_training=True,
          dependency_checks=checks,synthetic_report_only=True,main_rows=14,paired_rows=42,privacy_utility_rows=56,grid_heatmaps=14))
print('CLI dependency and synthetic-report validation passed; no training launched.')
