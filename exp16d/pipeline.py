"""Wait for exp16c, GPU acceptance, search, review, freeze, all formal trials, report."""
import os,time,traceback
from dataclasses import replace
from exp16d import RESULTS,ROOT
from exp16d.config import *
from exp16d.support import read,code_hashes
from exp16d.search import *
from exp16d.scheduler import run_batch

def phase(stage,**fields):save_json(RESULTS/'pipeline_status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),physical_gpus=[0,1,2,3],**fields))
def run():
    try:
        phase('waiting_for_exp16c',dependency=str(ROOT/'exp16c/results/pipeline_status.json'))
        while read(ROOT/'exp16c/results/pipeline_status.json')['stage']!='completed':
            phase('waiting_for_exp16c',predecessor=read(ROOT/'exp16c/results/pipeline_status.json'))
            time.sleep(30)
        from exp16d.stage1 import gpu_acceptance
        phase('gpu_acceptance');gpu_acceptance()
        jobs=additions();freeze_json('search_grid.json',dict(trials=[c.values() for c in jobs]));phase('search',planned=20)
        rows=run_batch('search',jobs);save_json(RESULTS/'search_results.json',rows)
        top=freeze_json('top2.json',top_two(rows))
        review=[replace(Trial(**v),stage='review',seed=s) for vs in top.values() for v in vs for s in RECHECK_SEEDS]
        extra=[c for c in controlled() if c.stage=='review']
        freeze_json('review_grid.json',dict(trials=list({c.id:c.values() for c in review+extra}.values())))
        phase('review',planned=16);reviews=run_batch('review',review);save_json(RESULTS/'review_results.json',reviews)
        winners,evidence=select(top,rows+reviews)
        freeze_json('selection_evidence.json',evidence);freeze_json('final_freeze.json',dict(winners=winners,code_sha256=code_hashes(),selection_sha256=file_hash(RESULTS/'selection_evidence.json')))
        phase('controlled');controls=run_batch('controlled',controlled());save_json(RESULTS/'controlled_results.json',controls)
        formal=[replace(Trial(**v),stage='final',seed=s) for v in winners['cv'].values() for s in FINAL_SEEDS]
        freeze_json('formal_grid.json',dict(trials=[c.values() for c in formal]));phase('formal',planned=40)
        save_json(RESULTS/'formal_results.json',run_batch('formal',formal))
        phase('report');from exp16d.report import report
        report();phase('completed',formal_completed=40)
    except Exception as e:
        phase('failed',error=str(e),traceback=traceback.format_exc());raise
if __name__=='__main__':run()
