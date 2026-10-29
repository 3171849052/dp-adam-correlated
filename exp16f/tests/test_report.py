"""Exercise complete reporting with fixture metrics; no training or selection."""
def test_report_all_outputs(tmp_path,monkeypatch):
    import json
    from exp16f import ROOT,RESULTS,history,support,report
    from exp16f.search import centers
    from exp16f.config import FINAL_SEEDS
    old=[dict(r,historical=True) for r in json.loads((ROOT/'exp16e/results/formal_results.json').read_text())]
    monkeypatch.setattr(history,'collect',lambda:old)
    monkeypatch.setattr(support,'verify_protected',lambda:None)
    monkeypatch.setattr(report,'RESULTS',tmp_path)
    from exp16f.matrices import load,audit_arrays
    from exp16f.config import file_hash
    matrices=[]
    for n in (250,310):
        for work,mechanism in [('identity','iid'),('sgd','b2lt'),('momentum','b2lt'),('momentum-bias','b2lt')]:
            a=load(work,mechanism,n)
            matrices.append(dict(total_steps=n,workload=work,mechanism=mechanism,sha256=file_hash(a['path']),**audit_arrays(a['A'],a['C'],a['D'])))
    (tmp_path/'matrix_audit.json').write_text(json.dumps(matrices))
    (tmp_path/'budget.json').write_text('{}')
    for name in ('search_results.json','selection_evidence.json','final_freeze.json'):(tmp_path/name).write_text('{}')
    winners={'cv':{},'nlp':{}};formal=[]
    for c in centers():
        winners[c.task][c.method]=c.values()
        for seed in FINAL_SEEDS:
            r=next(r for r in old if r['task']==c.task and r['seed']==seed and r['method']=='adam-mf-sgd-stp-b2lt')
            formal.append(dict(r,method=c.method,lr=c.lr,clip=c.clip,eps_scale=c.eps_scale,historical=False))
    report.generate(formal,dict(winners=winners))
    audit=json.loads((tmp_path/'final_audit.json').read_text())
    assert audit['new_count']==120 and audit['historical_count']==260
    assert len(audit['output_sha256'])==9
    import csv
    assert len(list(csv.DictReader((tmp_path/'formal_per_seed.csv').open())))==380
    assert len(list(csv.DictReader((tmp_path/'cv_summary.csv').open())))==19
