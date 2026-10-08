"""Automatic search -> immutable freeze -> 70 final runs -> report."""
import argparse
from exp7b import BASE
from exp7b.config import save_json
from exp7b.stage1 import verify_platform
from exp7b.search import search
from exp7b.final import run_final
from exp7b.audit import audit_fifo

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs=3, default=[1,2,3])
    args = parser.parse_args()
    assert sorted(args.gpus) == [1,2,3]
    with (BASE / 'results/stage2.log').open('a',buffering=1) as log:
        import contextlib
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            verify_platform(args.gpus)
            search(args.gpus)
            run_final(args.gpus)
            save_json(BASE / 'results/fifo_audit.json',audit_fifo())

if __name__ == '__main__':
    main()
