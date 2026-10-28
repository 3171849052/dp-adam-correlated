import json
from exp17b import report as reporting
from exp17b.common import save, R


def test_pending_and_completed_report(tmp_path,monkeypatch):
    monkeypatch.setattr(reporting,'RESULTS',tmp_path)
    monkeypatch.setattr(reporting,'output',lambda cfg:tmp_path/f"{cfg['stage']}_{cfg['method']}_{cfg['seed']}")
    reporting.report()
    assert '未完成' in (tmp_path/'report.md').read_text()
    results=[]
    for method in ['smooth','lc','gh']:
        for seed in [20261111,20261112]:
            cfg=dict(stage='D',method=method,alpha=0 if method=='smooth' else .5,r=R,lr=.001,seed=seed,steps=250)
            folder=reporting.output(cfg);folder.mkdir()
            d=dict(step=1,train_loss=1.,correction_norm=.1,update_norm=.2,direction_norm=2.,direction_cosine=.99,
                   relative_update_scale=1.1,correction_parallel=.01,correction_orthogonal=.09)
            (folder/'diagnostics.jsonl').write_text(json.dumps(d)+'\n')
            results.append(dict(config=cfg,accuracy=.7,validation_loss=1. if method=='smooth' else 1.1,
                                actual_epsilon=7.999,seconds=10.,peak_allocated_bytes=1000))
    save(tmp_path/'D_results.json',results)
    reporting.report()
    assert '未优于匹配 Smooth' in (tmp_path/'report.md').read_text()
    for name in ['final_results.csv','paired_differences.csv','diagnostics.csv','utility_summary.csv','training_curves.png']:
        assert (tmp_path/name).stat().st_size>0
