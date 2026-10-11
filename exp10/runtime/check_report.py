"""Synthetic report integration fixture only; not an experiment or real result."""
import csv,json
from pathlib import Path
from exp10 import BASE
from exp10.config import save_json,METHODS
from exp10.tests.test_workflow import synthetic_search
from exp10.search import top_two,recheck_jobs,choose_winners,formal_jobs
from exp10 import report
folder=BASE/'runtime/report_check';folder.mkdir(parents=True,exist_ok=True)
(folder/'SYNTHETIC_FIXTURE_ONLY.txt').write_text('All metrics here are synthetic testing fixtures; no trainings were executed. These are not Exp10 results.\n')
rows=synthetic_search();top2=top_two(rows)
reviews=[dict(j.values(),trial_id=j.id,status='completed',accuracy=.7,validation_loss=.6) for j in recheck_jobs(top2)]
winners,_=choose_winners(top2,rows,reviews)
save_json(folder/'frozen_configs.json',winners);save_json(folder/'search_results.json',rows)
save_json(folder/'initial_search_results.json',rows[:96])
formal=[]
for cfg in formal_jobs(winners):
    output=folder/'fixtures'/cfg.task/cfg.id;output.mkdir(parents=True,exist_ok=True)
    record=dict(cfg.values(),trial_id=cfg.id,result_dir=str(output.relative_to(folder)),
        accuracy=.7+(.01 if cfg.method==METHODS[0] else 0),validation_loss=.5,
        actual_epsilon=8.,seconds=1.,peak_allocated_bytes=100.)
    formal.append(record)
    with (output/'mechanism_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['preconditioner_change_relative','denominator_lag_relative'])
        writer.writeheader();writer.writerows(dict(preconditioner_change_relative=.01,denominator_lag_relative=.1 if cfg.method==METHODS[1] else 0) for _ in range(cfg.total_steps))
report.RESULTS=folder
report.generate_report(formal,dict(status='passed',historical_files_unchanged=True))
summary=json.loads((folder/'summary_statistics.json').read_text());paired=json.loads((folder/'paired_differences.json').read_text())
assert len(summary)==20 and all(s['n']==10 for s in summary)
assert len(paired)==4 and all(abs(p['mean']-.01)<1e-12 for p in paired if p['metric']=='accuracy')
for name in ['module_a_report.md','module_b_report.md','final_report.md','figures/mechanisms.png','figures/mechanisms.pdf','figures/search_trajectory.png']:
    assert (folder/name).stat().st_size>0
save_json(BASE/'results/report_generation_test.json',dict(status='passed',synthetic_fixture=True,full_trainings_started=0,
    metrics_per_method=5,formal_fixture_rows=40,summary_rows=20,paired_rows=4,reports=3,figures=3,
    output_directory=str(folder.relative_to(BASE)),privacy_budget_used=0))
print('Report integration passed with synthetic fixtures only; no training or official evaluation data used.')
