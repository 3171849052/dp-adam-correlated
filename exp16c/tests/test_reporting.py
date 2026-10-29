"""Synthetic fixtures only: exercise final report generation without real claims."""
import json
from dataclasses import replace
from exp16c.config import Trial,METHODS,FINAL_SEEDS
from exp16c.support import read
from exp16c import RESULTS

def test_report_artifacts_and_paired_statistics(tmp_path,monkeypatch):
    import exp16c.report as report
    import exp16c.stage1 as stage1
    monkeypatch.setattr(report,'RESULTS',tmp_path)
    monkeypatch.setattr(report,'verify',lambda cfg:None)
    monkeypatch.setattr(stage1,'verify_acceptance',lambda:None)
    monkeypatch.setattr(stage1,'verify_readonly',lambda:None)
    def write(name,value):(tmp_path/name).write_text(json.dumps(value))
    rows=[];winners={}
    for i,m in enumerate(METHODS):
        cfg=Trial('cv',m,.5,3.,3. if '-stp-' in m or m.startswith('stpex') else None)
        winners[m]=cfg.values()
        for seed in FINAL_SEEDS:
            c=replace(cfg,stage='final',seed=seed)
            rows.append(dict(**c.values(),trial_id=c.id,status='completed',accuracy=.6+i*.01+(seed-FINAL_SEEDS[0])*.001,validation_loss=1.,seconds=2.,train_seconds=1.,peak_allocated_bytes=100,peak_reserved_bytes=200,epochs=[dict(clip_fraction=.4)]*5,actual_epsilon=8.,initialization_sha256=str(seed),classifier_initialization_sha256=str(seed),train_order_sha256=str(seed),rng_trace=[seed]))
    write('formal_results.json',rows);write('final_freeze.json',dict(winners={'cv':winners}))
    write('matrix_audit.json',read(RESULTS/'matrix_audit.json'));write('selection_evidence.json',{})
    for stage in ('search','review','formal'):
        write(stage+'_status.json',{})
        if stage!='formal':write(stage+'_results.json',[])
    report.report()
    assert read(tmp_path/'final_audit.json')['formal_completed']==90
    import csv
    with (tmp_path/'summary.csv').open() as f:summary=list(csv.DictReader(f))
    assert len(summary)==9 and abs(float(summary[0]['accuracy_mean'])-60.45)<1e-10
    with (tmp_path/'paired_differences.csv').open() as f:pairs=list(csv.DictReader(f))
    assert len(pairs)==72 and abs(float(pairs[0]['mean'])-1.)<1e-10
    assert (tmp_path/'final_report.md').exists()
