from dataclasses import replace
import pytest
from exp16c.config import Trial,A_METHODS,RECHECK_SEEDS
from exp16c.search import select,rank

def records(c,acc,loss=1.):
    return [dict(**x.values(),trial_id=x.id,status='completed',accuracy=acc,validation_loss=loss,official_evaluation_used=False) for x in [c,*[replace(c,stage='review',seed=s) for s in RECHECK_SEEDS]]]

def test_incumbent_survives_search_rank_and_three_seed_tie():
    m=A_METHODS[0];old=Trial('cv',m,.003,17.3205080757);new=Trial('cv',m,.003,12.);other=Trial('cv',m,.003,24.)
    rows=records(old,.8)+records(new,.79)+records(other,.78)
    rows[3]['accuracy']=.82
    top={m:[new.values(),other.values()]}
    win,evidence=select(top,rows,{m:old.values()})
    assert win['cv'][m]==old.values() and evidence[m]['incumbent_retained']
    rows=records(old,.8)+records(new,.8)+records(other,.78)
    assert select(top,rows,{m:old.values()})[0]['cv'][m]==old.values()
    rows=records(old,.8)+records(new,.81)+records(other,.78)
    assert select(top,rows,{m:old.values()})[0]['cv'][m]==new.values()
    rows[4]['status']='non_finite'
    assert select(top,rows,{m:old.values()})[0]['cv'][m]==old.values()

def test_loss_tiebreak_and_no_official_selection():
    c=Trial('cv',A_METHODS[0],.003,12.);d=replace(c,clip=24.)
    rows=records(c,.8,1.)+records(d,.8,.9)
    assert rank([rows[0],rows[3]])[0]['clip']==24.
    rows[0]['official_evaluation_used']=True
    with pytest.raises(ValueError):rank(rows)
