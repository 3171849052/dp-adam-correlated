"""Detached CV search -> reviews -> freeze -> actual formal training -> report."""
import os,time,traceback
from dataclasses import replace
from exp16c import RESULTS,ROOT
from exp16c.config import *
from exp16c.support import read,code_hashes
from exp16c.search import additions,top_two,freeze_json,select
from exp16c.scheduler import run_batch
from exp16c import history

def phase(stage,**fields):save_json(RESULTS/'pipeline_status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**fields))
def run():
    from exp16c.stage1 import verify_acceptance
    verify_acceptance()
    try:
        phase('historical_verification')
        old_search=history.rows('search');old_review=history.rows('review');incumbents=history.winners()
        if len([r for r in old_search if r['method']==STPEX])!=27:raise ValueError('Expected 27 completed STP-EX searches')
        freeze_json('historical_search.json',old_search);freeze_json('historical_review.json',old_review)
        old={r['trial_id']:r for r in old_search+old_review}
        jobs=[c for c in additions() if c.id not in old]
        freeze_json('search_grid.json',dict(trials=[c.values() for c in jobs]));phase('search',new_trials=len(jobs))
        new_search=run_batch('search',jobs)
        merged=list({r['trial_id']:r for r in old_search+new_search}.values());save_json(RESULTS/'search_results.json',merged)
        top=freeze_json('top2.json',top_two(merged))
        review=[replace(Trial(**v),stage='review',seed=s) for vs in top.values() for v in vs for s in RECHECK_SEEDS]
        jobs=[c for c in review if c.id not in old]
        freeze_json('review_grid.json',dict(trials=[c.values() for c in jobs]));phase('review',new_trials=len(jobs))
        reviewed=run_batch('review',jobs);save_json(RESULTS/'review_results.json',old_review+reviewed)
        winners,evidence=select(top,merged+old_review+reviewed,incumbents)
        # Untouched methods retain their original selection evidence.
        original=read(ROOT/'exp16a/results/selection_evidence.json')['cv']
        for m in HISTORICAL_METHODS:
            if m not in evidence:evidence[m]=dict(historical_evidence=original[m],selected_config=incumbents[m])
        freeze_json('selection_evidence.json',evidence)
        freeze_json('final_freeze.json',dict(winners=winners,code_sha256=code_hashes(),selection_sha256=file_hash(RESULTS/'selection_evidence.json')))
        formal=[replace(Trial(**v),stage='final',seed=s) for v in winners['cv'].values() for s in FINAL_SEEDS]
        reused=[];jobs=[]
        for c in formal:
            r=history.formal(c) if c.method in HISTORICAL_METHODS else None
            if r is None:jobs.append(c)
            else:reused.append(r)
        freeze_json('formal_grid.json',dict(trials=[c.values() for c in jobs],historical_trial_ids=[r['trial_id'] for r in reused]))
        phase('formal',new_trials=len(jobs),reused=len(reused),total=len(formal))
        results=run_batch('formal',jobs);save_json(RESULTS/'formal_results.json',reused+results)
        phase('report');from exp16c.report import report
        report();phase('completed',formal_completed=len(formal),new_formal_completed=len(results))
    except Exception as error:
        phase('failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        save_json(RESULTS/'failure.json',read(RESULTS/'pipeline_status.json'));raise
if __name__=='__main__':run()
