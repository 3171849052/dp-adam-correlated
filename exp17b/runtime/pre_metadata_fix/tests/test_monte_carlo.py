import csv
import numpy as np
from exp17b.common import PROTOCOL, save, sha
from exp17b import monte_carlo as mc


def test_small_mc_reproducible_and_risk_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(mc,'RESULTS',tmp_path)
    protocol=dict(PROTOCOL,mc=dict(PROTOCOL['mc'],histories=2,samples=1000,t=[1,2],snr=[0,1]))
    monkeypatch.setattr(mc,'PROTOCOL',protocol)
    save(tmp_path/'protocol_freeze.json',{'calibration':{'search':{'nu_squared':.01}}})
    mc.run()
    with (tmp_path/'mse_summary.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==2*2*2*4*(5+9+9)
    for row in rows:
        assert np.isclose(float(row['mse']),float(row['bias2'])+float(row['variance']),rtol=1e-10,atol=1e-14)
        if float(row['alpha'])==0:assert float(row['mse_ratio'])==1
        assert not (row['method']=='lc' and float(row['r'])==0)
    before=sha(tmp_path/'mse_summary.csv')
    (tmp_path/'mc_complete.json').unlink();mc.run()
    assert sha(tmp_path/'mse_summary.csv')==before
