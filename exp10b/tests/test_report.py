import json
import torch
from exp10b.config import save_json
from exp10b.protocol import choose_winners,formal_jobs
from exp10b.tests.test_workflow import internal_rows

def test_complete_report_pipeline_with_isolated_fixtures(tmp_path,monkeypatch):
    from exp10b import report
    monkeypatch.setattr(report,'RESULTS',tmp_path)
    monkeypatch.setattr(report,'historical_cv_reproduction',lambda rows:[])
    internal=internal_rows();winners,evidence=choose_winners(internal)
    final=[dict(j.values(),accuracy=.7+(.01 if j.module>0 else 0),validation_loss=.5) for j in formal_jobs(winners)]
    for rows in (internal,final):
        for i,row in enumerate(rows):
            row.update(seconds=1.,peak_allocated_bytes=100.,peak_reserved_bytes=200.,result_dir=f"{row['stage']}/{row['task']}/{i}")
            directory=tmp_path/row['result_dir'];directory.mkdir(parents=True)
            (directory/'mechanism_metrics.csv').write_text('preconditioner_mean,preconditioner_std,preconditioner_change_relative\n1,2,0\n2,3,1\n')
    report.search_report(internal,evidence)
    report.final_report(internal,final,evidence,winners,{'status':'passed'})
    assert all((tmp_path/'figures'/name).stat().st_size>0 for name in (
        'lambda_accuracy.png','j1_j2_pareto.png','j2_decomposition.png','final_paired_differences.png'))
    assert '160 full' in (tmp_path/'report.md').read_text()
    assert 'not new blind' in (tmp_path/'report.md').read_text()
    assert len(json.loads((tmp_path/'paired_differences.json').read_text()))==10
    assert len(json.loads((tmp_path/'matrix_diagnostics.json').read_text()))==18

def test_historical_reproduction_records_exact_and_differences(tmp_path,monkeypatch):
    from exp10b import report
    monkeypatch.setattr(report,'ROOT',tmp_path);monkeypatch.setattr(report,'RESULTS',tmp_path/'new')
    directory=tmp_path/'exp9/results/final/cv/reference';directory.mkdir(parents=True)
    historical=dict(status='completed',method='dp-adam-momentum-bias-bandinvmf',lr=.003,C=30,
        seed=20261111,planned_total_steps=250,initialization_sha256='init',classifier_initialization_sha256='head',
        train_order_sha256='order',rng_trace=['rng'],strategy_sha256='matrix',coefficients=[1.,-.5,-.1,-.3],
        calibration=dict(target_mu=1.,sensitivity=2.,innovation_std_sum=60.),accuracy=.7,
        epochs=[dict(validation_loss=.5)])
    save_json(directory/'summary.json',historical)
    row=dict(historical,task='cv',module=0.,innovation_seed=20261112,result_dir='final/cv/new',validation_loss=.5)
    new=tmp_path/'new'/row['result_dir'];new.mkdir(parents=True)
    checkpoint=dict(model={'weight':torch.tensor([1.,2.])})
    torch.save(checkpoint,directory/'checkpoint.pt');torch.save(checkpoint,new/'checkpoint.pt')
    assert report.historical_cv_reproduction([row])[0]['status']=='exact_reproduction'
    torch.save(dict(model={'weight':torch.tensor([1.,2.1])}),new/'checkpoint.pt')
    comparison=report.historical_cv_reproduction([row])[0]
    assert comparison['status']=='difference_recorded' and comparison['final_model_max_abs_difference']>0
