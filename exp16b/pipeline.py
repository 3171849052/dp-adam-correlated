from exp16b import RESULTS
from exp16b.config import save_json
from exp16b.stage1 import verify_acceptance
from exp16b.search import coarse_jobs
from exp16b.scheduler import run_batch
import traceback,time

def run():
    verify_acceptance()
    save_json(RESULTS/'pipeline_status.json',dict(status='running',stage='coarse',started=time.time()))
    try:
        rows=run_batch('coarse',coarse_jobs())
        save_json(RESULTS/'coarse_results.json',rows)
        save_json(RESULTS/'pipeline_status.json',dict(status='completed',stage='coarse',ended=time.time()))
    except Exception:
        save_json(RESULTS/'pipeline_status.json',dict(status='blocked',traceback=traceback.format_exc(),ended=time.time()))
        raise
