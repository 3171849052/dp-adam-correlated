"""Exactly 70 full runs, using only immutable frozen configurations."""
import argparse
from exp7b.config import METHODS, FINAL_SEEDS, trial
from exp7b.frozen import load_frozen
from exp7b.launcher import run_queue
from exp7b.report import final_report

def final_jobs():
    frozen = load_frozen()
    jobs = []
    for i, seed in enumerate(FINAL_SEEDS):
        # Change the leading method by seed to spread physical GPU assignments.
        methods = list(METHODS)
        for method in methods[i%7:] + methods[:i%7]:
            row = frozen[method]
            jobs.append(trial(method,row['lr'],row['C'],row['eps_scale'],seed))
    assert len(jobs) == 70
    return jobs

def run_final(gpus):
    run_queue(final_jobs(),gpus,category='final')
    final_report()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs=3, default=[1,2,3])
    run_final(parser.parse_args().gpus)
