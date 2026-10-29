"""Single-instance offline experiment, restartable at sealed trial boundaries."""
import time,traceback
from dataclasses import replace
from exp16e import RESULTS
from exp16e.config import save_json,FINAL_SEEDS,Trial
from exp16e.support import read,code_hashes
from exp16e import history,search
from exp16e.scheduler import run_batch

def status(phase,**extra):save_json(RESULTS/'progress.json',dict(phase=phase,time=time.time(),**extra))
def local(rows):return [dict(r,source=str(history.config(r).output.relative_to(history.ROOT)),historical=False) for r in rows]

def main():
    try:
        acceptance=read(RESULTS/'acceptance.json');assert acceptance['status']=='passed' and acceptance['code_sha256']==code_hashes()
        status('verifying_history')
        historical=history.collect('search')+history.collect('review')
        search.freeze_json('historical_internal_results.json',historical)
        jobs=search.additions();search.freeze_json('search_grid.json',dict(trials=[c.values() for c in jobs]))
        idx={history.key(r):r for r in historical}
        missing=[c for c in jobs if history.key(c) not in idx]
        save_json(RESULTS/'budget.json',dict(requested_search=67,new_search=len(missing),fixed_review_max=32,conditional_review_max=16,necessary_formal_max=80,conditional_formal_max=80,total_upper_bound=275))
        status('search',planned=len(missing));new=local(run_batch('search',missing))
        all_rows=list({history.key(r):r for r in historical+new}.values());save_json(RESULTS/'search_results.json',all_rows)
        chosen,decisions,reviews=search.plan_review(all_rows)
        search.freeze_json('review_plan.json',dict(chosen=chosen,decisions=decisions))
        search.freeze_json('review_grid.json',dict(trials=[c.values() for c in reviews]))
        idx={history.key(r):r for r in all_rows};missing_reviews=[c for c in reviews if history.key(c) not in idx]
        status('review',planned=len(missing_reviews));review_rows=local(run_batch('review',missing_reviews))
        all_rows=list({history.key(r):r for r in all_rows+review_rows}.values());save_json(RESULTS/'search_results.json',all_rows)
        winners,evidence=search.select(chosen,decisions,all_rows)
        search.freeze_json('selection_evidence.json',evidence)
        freeze=search.freeze_json('final_freeze.json',dict(winners=winners,code_sha256=code_hashes(),selection='three-seed internal accuracy mean descending, cross entropy ascending, canonical ID'))
        status('frozen')
        from exp16e.assets import prepare
        prepare(official=True)
        formal_jobs=[replace(Trial(**v),stage='final',seed=s) for methods in winners.values() for v in methods.values() for s in FINAL_SEEDS]
        reused=[];missing_formal=[]
        for c in formal_jobs:
            old=history.formal(c)
            if old is None:missing_formal.append(c)
            else:reused.append(old)
        search.freeze_json('formal_grid.json',dict(trials=[c.values() for c in missing_formal],historical=[dict(source=r['source'],trial_id=r['trial_id'],summary_sha256=r['summary_sha256']) for r in reused]))
        budget=read(RESULTS/'budget.json');budget.update(new_review=len(missing_reviews),new_formal=len(missing_formal),historical_formal=len(reused),planned_new_total=len(missing)+len(missing_reviews)+len(missing_formal))
        assert budget['planned_new_total']<=275 and len(missing_formal)<=160
        save_json(RESULTS/'budget.json',budget)
        status('formal',planned_new=len(missing_formal),historical=len(reused));formal=reused+local(run_batch('formal',missing_formal))
        save_json(RESULTS/'formal_results.json',formal);status('report')
        from exp16e.report import generate
        generate(formal,freeze);status('completed',formal_count=len(formal))
    except BaseException as e:
        status('failed',error=str(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
