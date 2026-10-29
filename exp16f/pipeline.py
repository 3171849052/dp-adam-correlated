"""Restartable full experiment; official evaluation follows configuration freeze."""
import time,traceback
from dataclasses import replace
from exp16f import RESULTS
from exp16f.config import *
from exp16f.support import read,code_hashes
from exp16f import search,history
from exp16f.scheduler import run_batch

def status(phase,**extra):save_json(RESULTS/'progress.json',dict(phase=phase,time=time.time(),**extra))
def local(rows):return [dict(r,source=str(history.config(r).output.relative_to(history.ROOT)),historical=False) for r in rows]
def main():
    try:
        a=read(RESULTS/'acceptance.json');assert a['status']=='passed' and a['code_sha256']==code_hashes()
        jobs=search.additions();search.freeze_json('search_grid.json',dict(trials=[c.values() for c in jobs]))
        status('search',planned=132);rows=local(run_batch('search',jobs))
        extra,decisions=search.boundary(rows);assert len(extra)<=24
        search.freeze_json('boundary_grid.json',dict(trials=[c.values() for c in extra],decisions=decisions))
        status('boundary',planned=len(extra));rows+=local(run_batch('boundary',extra));save_json(RESULTS/'search_results.json',rows)
        chosen,reviews=search.plan_review(rows);assert len(reviews)==48
        search.freeze_json('review_grid.json',dict(trials=[c.values() for c in reviews],chosen=chosen))
        status('review',planned=48);rows+=local(run_batch('review',reviews));save_json(RESULTS/'search_results.json',rows)
        winners,evidence=search.select(chosen,rows);search.freeze_json('selection_evidence.json',evidence)
        freeze=search.freeze_json('final_freeze.json',dict(winners=winners,code_sha256=code_hashes(),selection='three-seed internal accuracy descending, cross entropy ascending, canonical ID'))
        from exp16f.assets import prepare
        prepare(official=True)
        jobs=[replace(Trial(**v),stage='final',seed=s) for ms in winners.values() for v in ms.values() for s in FINAL_SEEDS]
        assert len(jobs)==120
        save_json(RESULTS/'budget.json',dict(search=132,boundary=len(extra),review=48,formal=120,planned_new_total=300+len(extra),maximum=324))
        status('formal',planned=120);formal=local(run_batch('formal',jobs))
        save_json(RESULTS/'formal_results.json',formal);status('report')
        from exp16f.report import generate
        generate(formal,freeze);status('completed',new_formal_count=120)
    except BaseException as e:
        status('failed',error=str(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
