from dataclasses import replace
import pytest
from exp16f import search
from exp16f.config import Trial
from exp16f.runtime.recover import candidates

def row(c,accuracy=.7):return dict(**c.values(),trial_id=c.id,status='completed',accuracy=accuracy,validation_loss=1.,official_evaluation_used=False)
def failure(c):return dict(**c.values(),trial_id=c.id,status='non_finite',error='FloatingPointError: Nonfinite Adam state')
def test_numerical_failure_missing_evaluation_field():
    c=search.centers()[0];a=row(c);bad=failure(replace(c,clip=c.clip*3))
    assert candidates([bad,a],c)==[a]

@pytest.mark.parametrize('kind',['official','official_failure','wrong_stage','unknown_failure','missing_completed_field'])
def test_reject_invalid_selection_input(kind):
    c=search.centers()[0];r=row(c)
    if kind=='official':r['official_evaluation_used']=True
    if kind=='official_failure':r=failure(c);r['official_evaluation_used']=True
    if kind=='wrong_stage':r['stage']='final'
    if kind=='unknown_failure':r['status']='oom'
    if kind=='missing_completed_field':del r['official_evaluation_used']
    with pytest.raises((AssertionError,KeyError)):candidates([r],c)

def test_boundary_and_review_keep_successful_ranking(monkeypatch):
    rows=[row(c,c.lr) for c in search.additions()]
    original_boundary=search.boundary(rows);original_review=search.plan_review(rows)
    monkeypatch.setattr(search,'candidates',candidates)
    rows.append(failure(replace(search.centers()[0],lr=99)))
    assert search.boundary(rows)==original_boundary
    assert search.plan_review(rows)==original_review
    assert len(original_review[1])==48
