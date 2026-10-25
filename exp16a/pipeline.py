"""Coarse -> one fine grid -> Top-2 reviews -> 28 winners -> 280 formal -> audit."""
import os,time,traceback
from dataclasses import replace
from exp16a import RESULTS
from exp16a.config import Trial,save_json,file_hash,FINAL_SEEDS
from exp16a.support import read,code_hashes,verify_protected
from exp16a.search import coarse_jobs,fine_jobs,top_two,review_jobs,select,freeze_json
from exp16a.scheduler import run_batch,verify

def phase(name,**fields):save_json(RESULTS/'pipeline_status.json',dict(stage=name,pid=os.getpid(),updated=time.time(),**fields))
def run():
    from exp16a.stage1 import verify_acceptance
    verify_acceptance();verify_protected()
    try:
        phase('coarse');coarse=coarse_jobs();freeze_json('coarse_grid.json',dict(trials=[c.values() for c in coarse]))
        rows=run_batch('coarse',coarse);save_json(RESULTS/'coarse_results.json',rows)
        phase('fine');fine,basis=fine_jobs(rows)
        freeze_json('fine_grid.json',dict(trials=[c.values() for c in fine],basis=basis,rounds=1))
        fine_rows=run_batch('fine',fine);save_json(RESULTS/'fine_results.json',fine_rows)
        merged=list({r['trial_id']:r for r in rows+fine_rows}.values());save_json(RESULTS/'search_results.json',merged)
        top=freeze_json('top2.json',top_two(merged));review=review_jobs(top)
        freeze_json('review_grid.json',dict(trials=[c.values() for c in review]));phase('review')
        reviewed=run_batch('review',review);save_json(RESULTS/'review_results.json',reviewed)
        winners,evidence=select(top,merged,reviewed);freeze_json('selection_evidence.json',evidence)
        manifest=dict(winners=winners,code_sha256=code_hashes(),phase1_sha256=file_hash(RESULTS/'phase1_acceptance.json'),
            selection_sha256=file_hash(RESULTS/'selection_evidence.json'),evidence_sha256={name:file_hash(RESULTS/name) for name in ('coarse_grid.json','fine_grid.json','review_grid.json','search_results.json','review_results.json')})
        freeze_json('final_freeze.json',manifest)
        from exp16a.support import official_assets
        freeze_json('official_assets.json',official_assets())
        final=[replace(Trial(**winners[t][m]),stage='final',seed=s) for t in winners for m in winners[t] for s in FINAL_SEEDS]
        if len(final)!=280:raise ValueError('Formal count must be 280')
        freeze_json('formal_grid.json',dict(trials=[c.values() for c in final]));phase('formal',frozen_configurations=28)
        formal=run_batch('formal',final);save_json(RESULTS/'formal_results.json',formal)
        phase('audit')
        for c in {c.id:c for c in coarse+fine+review+final}.values():verify(c)
        from exp16a.report import report
        report();verify_protected();phase('completed',formal_completed=280)
    except Exception as error:
        phase('failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc());raise
if __name__=='__main__':run()
