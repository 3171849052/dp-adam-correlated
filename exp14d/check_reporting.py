"""Exercise report generation with explicit synthetic fixtures, never production results."""
import json
from exp14d import BASE
from exp14d.config import METHODS, FINAL_SEEDS, save_json
from exp14d.support import write_csv

def main():
    import exp14d.report as report
    target=BASE/'runtime/report_fixture';target.mkdir(parents=True,exist_ok=True)
    rows=[]
    for task in ('cv','nlp'):
        for method in METHODS:
            for seed in FINAL_SEEDS:
                d=target/f'{task}_{method}_{seed}';d.mkdir(exist_ok=True)
                row=dict(task=task,method=method,seed=seed,accuracy=.5+(seed-FINAL_SEEDS[0])*.001,lr=.001,C=1,eps_scale=None,source_summary=str(d/'summary.json'),epochs=[dict(accuracy=.4+i*.02) for i in range(5)])
                rows.append(row);save_json(d/'summary.json',row);save_json(d/'config.json',{})
                write_csv(d/'metrics.csv',[dict(epoch=1,accuracy=.5)])
                write_csv(d/'mechanism_metrics.csv',[dict(step=i,clip_fraction=.9) for i in (1,2)])
                (d/'optimizer_metrics.jsonl').write_text('\n'.join(json.dumps(dict(step=i,update_norm=.01,v_mean=.02,vhat_mean=.03,lr=.001)) for i in (1,2))+'\n')
    save_json(target/'A_formal_results.json',rows);save_json(target/'B_formal_results.json',rows);save_json(target/'A_config_changes.json',[])
    report.RESULTS=target;report.historical_rows=lambda *args:rows
    report.main()
    assert len(list((target/'figures').glob('*.png')))==4
    assert (target/'final_report.md').stat().st_size>1000
    save_json(BASE/'results/reporting_test.json',dict(status='passed',synthetic_fixture=True,figures=4))
    print('Report tables, paired statistics, diagnostics and figures PASS')
if __name__=='__main__':main()
