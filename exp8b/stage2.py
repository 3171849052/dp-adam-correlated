"""Stage 1 gate → single-seed search → immutable freeze → 70 finals → report."""
import argparse
import traceback
from exp8b import RESULTS
from exp8b.audit import check_platform
from exp8b.config import save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--gpus',type=int,nargs='+')
    p.add_argument('--per-gpu',type=int,choices=(1,2),default=2)
    a=p.parse_args()
    queue=None
    try:
        check_platform()
        # Re-run correctness tests before any long search/final work.
        from exp8b.stage1 import unit_tests
        unit_tests(prefix="stage2_")
        from exp8b.search import search
        from exp8b.final import final
        if a.gpus:
            from exp8b.launcher import MultiGPUQueue
            queue=MultiGPUQueue(a.gpus,a.per_gpu)
        if not (RESULTS/'frozen_configs.json').exists():
            save_json(RESULTS/'stage2_status.json',dict(status='running',phase='search',gpu=a.gpu))
            search(a.gpu,queue=queue) if queue else search(a.gpu)
        save_json(RESULTS/'stage2_status.json',dict(status='running',phase='final',gpu=a.gpu))
        final(a.gpu,queue=queue) if queue else final(a.gpu)
        save_json(RESULTS/'stage2_status.json',dict(status='completed',final_trials=70,gpu=a.gpu))
    except Exception as error:
        save_json(RESULTS/'stage2_status.json',dict(status='failed',error=str(error),traceback=traceback.format_exc(),gpu=a.gpu))
        raise
    finally:
        if queue: queue.close()

if __name__=='__main__': main()
