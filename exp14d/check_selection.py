"""Verify merged-history selection and both formal-grid cardinalities."""
import json
from exp14d import BASE, ROOT
from exp14d.config import save_json, FINAL_SEEDS
from exp14d.support import historical_rows, boundary_grid, grid
from exp14d import pipeline

def main():
    target=BASE/'runtime/selection_fixture';target.mkdir(parents=True,exist_ok=True)
    pipeline.RESULTS=target
    old=historical_rows('exp14c','search')
    original=json.loads((ROOT/'exp14c/results/best_configs.json').read_text())
    for o in original:
        winner=pipeline.rank([r for r in old if r['task']==o['task'] and r['method']==o['method']+'-adam3'])[0]
        assert winner['trial_id']==o['trial_id']
    def fixture(c,acc):
        d=target/c.id;d.mkdir(exist_ok=True)
        r=dict(c.values(),trial_id=c.id,status='completed',accuracy=acc,epochs=[dict(validation_loss=.5)]*5,official_evaluation_used=False,source_summary=str(d/'summary.json'))
        for n in ('summary.json','config.json','metrics.csv'):save_json(d/n,r)
        return r
    for acc,expected in ((0.,0),(1.,4)):
        winners=pipeline.freeze(old+[fixture(c,acc) for c in boundary_grid()],'A')
        changed=[w for w in winners if any(o['task']==w['task'] and o['method']+'-adam3'==w['method'] and (o['lr'],o['C'])!=(w['lr'],w['C']) for o in original)]
        assert len(changed)==expected
        assert len(pipeline.finals(changed,'A'))==10*expected
    winners=pipeline.freeze([fixture(c,.5) for task in ('cv','nlp') for c in grid(task)],'B')
    final=pipeline.finals(winners,'B')
    assert len(final)==len({c.id for c in final})==120
    for w in winners:assert sorted(c.seed for c in final if (c.task,c.method)==(w['task'],w['method']))==list(FINAL_SEEDS)
    save_json(BASE/'results/selection_test.json',dict(status='passed',old_winners_exact=True,A_formal_bounds=[0,40],B_formal=120))
    print('Historical winners, A 0/40 transition, B 120 formal grid PASS')
if __name__=='__main__':main()
