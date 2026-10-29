from dataclasses import replace
import pytest
from exp16d.config import *
from exp16d.search import additions,top_two,select
from exp16d.report import stats,differences

def test_accuracy_ties_use_loss_at_both_selection_stages():
    rows=[dict(**c.values(),trial_id=c.id,status='completed',accuracy=.5,validation_loss=float(i+1),official_evaluation_used=False) for i,c in enumerate(additions())]
    top=top_two(rows)
    for vs in top.values():
        for i,v in enumerate(vs):
            for seed in RECHECK_SEEDS:
                c=replace(Trial(**v),stage='review',seed=seed)
                rows.append(dict(**c.values(),trial_id=c.id,status='completed',accuracy=.5,validation_loss=10.-i*5,official_evaluation_used=False))
    winners,evidence=select(top,rows)
    assert all(winners['cv'][m]==vs[1] for m,vs in top.items())

def test_paired_statistics_and_rng_mismatch():
    def row(seed,accuracy):return dict(seed=seed,accuracy=accuracy,validation_loss=1.,initialization_sha256='x',classifier_initialization_sha256='y',train_order_sha256='z',rng_trace=['a'])
    g={'a':[row(s,.5) for s in FINAL_SEEDS],'b':[row(s,.6) for s in FINAL_SEEDS]}
    p=differences(g,['a','b']);assert p[0]['mean']==pytest.approx(10)
    assert p[0]['ci95_low']==pytest.approx(10)
    g['b'][0]['rng_trace']=['changed']
    with pytest.raises(ValueError):differences(g,['a','b'])
    assert stats([1,2,3])['sd']==1
