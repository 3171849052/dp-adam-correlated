"""Audit real completed trials and verify full workflow reuse without GPU trials."""
from exp5.runtime import EXP, ROOT
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from exp5.config import METHODS, FINAL_SEEDS, Trial
from exp5.launch_batch import read_completed
from exp5.train import write_json


def audit():
    search = json.loads((EXP / 'results/search/search_summary.json').read_text())
    assert search['status'] == 'completed' and len(search['stages']) == 5
    for row in search['trials']:
        read_completed(Trial(**{k: row[k] for k in ('method', 'seed', 'lr', 'tau', 'C')}), row['source_trial'])
    final = {}
    for path in (EXP / 'results/final').glob('*/summary.json'):
        row = json.loads(path.read_text())
        trial = Trial(**{k: row[k] for k in ('method', 'seed', 'lr', 'tau', 'C')})
        read_completed(trial, path.parent)
        with (path.parent / 'steps.csv').open() as stream:
            steps = list(csv.DictReader(stream))
        assert len(steps) == 250 and [int(r['step']) for r in steps] == list(range(1, 251))
        assert all(float(r['per_step_sensitivity']) == .002 for r in steps)
        assert row['privacy']['accountant_source'] == 'exp2.privacy.fixed_epoch_sensitivity'
        final[trial.method, trial.seed] = (row, steps[0])
    assert len(final) == 6
    for seed in FINAL_SEEDS:
        a, ar = final[METHODS[0], seed]
        b, br = final[METHODS[1], seed]
        assert a['initialization_sha256'] == b['initialization_sha256']
        for key in ('train_loss', 'train_top1', 'mean_raw_sample_norm', 'mean_transformed_sample_norm',
                    'clip_fraction', 'mean_clip_factor', 'raw_mean_gradient_norm', 'query_norm', 'batch_coherence'):
            assert math.isclose(float(ar[key]), float(br[key]), rel_tol=1e-6, abs_tol=1e-8), (seed, key)
    paths = [p for category in ('search/trials', 'final')
             for folder in (EXP / 'results' / category).iterdir() if folder.is_dir()
             for p in folder.iterdir() if p.name in ('train.log', 'steps.csv', 'metrics.csv', 'summary.json', 'resume.pt', 'final.pt')]
    before = {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in paths}
    original_report = (EXP / 'results/final/final_summary.json').read_text()
    with (EXP / 'results/resume_verification.log').open('w') as stream:
        subprocess.run([sys.executable, '-B', '-m', 'exp5.search', '--gpus', '0,1,2', '--run-finals'],
                       cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2'),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    after = {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in paths}
    assert before == after, 'Completed trial files changed during reuse'
    assert original_report == (EXP / 'results/final/final_summary.json').read_text()
    write_json(EXP / 'results/final_verification.json',
               dict(status='passed', completed_search_trials=len(search['trials']), final_runs=6,
                    paired_initialization_and_first_query=True, per_step_sensitivity=.002,
                    steps_per_run=250, physical_batches_per_run=2500,
                    resume_reused_all_trials=True, completed_trial_files_unchanged=True,
                    final_report_unchanged=True))
    print('Real-run audit and completed-workflow reuse passed.', flush=True)

if __name__ == '__main__':
    audit()
