from exp17c.pipeline import configurations,ranking
from exp17c.common import PROTOCOL

def test_counts_and_new_seeds():
    assert len(configurations('B',PROTOCOL['alphas']))==30
    assert len(configurations('C',[0,1e-14,1e-13,1e-12]))==20
    assert len(configurations('D',[0,1e-14]))==20
    sets=[set(PROTOCOL[s]['seeds']) for s in 'BCD']
    assert not sets[0]&sets[1] and not sets[1]&sets[2] and not sets[0]&sets[2]

def test_ce_primary():
    rows=[dict(config=dict(alpha=a),validation_loss=loss,accuracy=acc) for a,loss,acc in [(0,2,.9),(1e-14,1,.1),(1e-13,1,.2)]]
    assert ranking(rows)==[1e-13,1e-14,0]
