"""User-requested background formal queue; preserve the frozen training sources."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from exp12 import BASE, RESULTS
from exp12.config import Trial, METHOD, FINAL_SEEDS, save_json, file_hash
from exp12.frozen import load_frozen
from exp12.launcher import run_queue
from exp12.audit import audit_trial, code_hashes, protected_snapshot, scheduler_audit
from exp12.report import generate, write_csv


def execute(gpus):
    assert gpus == [3]
    assert os.environ.get('CONDA_DEFAULT_ENV') == 'curve'
    assert not (RESULTS/'final_gpu3_launch.json').exists(), 'Inspect existing launch; no automatic retry'
    winner = load_frozen()
    previous = json.loads((RESULTS/'run_state.json').read_text())
    assert previous['status'] == 'stopped_by_user'
    jobs = [Trial('cv', METHOD, winner['lr'], winner['C'], seed=s, stage='final') for s in FINAL_SEEDS]
    # No checkpoint/RNG state was saved mid-training: restart the interrupted seeds from step zero.
    archive = RESULTS/'interrupted'/'initial_final'
    archive.mkdir(parents=True)
    mapping = {}
    for name in ('run_state.json', 'progress.json', 'report.md', 'statistics.json',
                 'workflow.log', 'scheduler.jsonl', 'gpu_telemetry.jsonl'):
        path = RESULTS/name
        if path.exists():
            target = archive/'state'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            mapping[name] = dict(archive=str(target.relative_to(RESULTS)), sha256=file_hash(target))
    for job in jobs:
        if job.output.exists():
            assert (job.output/'stopped.json').exists() and not (job.output/'summary.json').exists()
            target = archive/'cv'/job.id
            target.parent.mkdir(parents=True, exist_ok=True)
            for path in sorted(job.output.rglob('*')):
                if path.is_file():
                    key=str(path.relative_to(RESULTS))
                    mapping[key]=dict(archive=str((target/path.relative_to(job.output)).relative_to(RESULTS)),
                                      sha256=file_hash(path))
            shutil.move(str(job.output), str(target))
    save_json(archive/'archive_manifest.json', mapping)
    started = time.time()
    save_json(RESULTS/'final_gpu3_launch.json', dict(status='running', pid=os.getpid(), started_unix=started,
              physical_gpus=gpus, max_workers_per_gpu=1, seeds=list(FINAL_SEEDS), lr=winner['lr'], C=winner['C'],
              frozen_manifest_sha256=file_hash(RESULTS/'frozen_manifest.json'),
              training_code_sha256=code_hashes(), runner_sha256=file_hash(Path(__file__)),
              restart_interrupted_seeds_from_step_zero=True, archive='interrupted/initial_final',
              user_requested_background=True, automatic_retry=False))
    save_json(RESULTS/'protected_files_before_gpu3.json', protected_snapshot())
    save_json(RESULTS/'run_state.json', dict(status='running', stage='final', pid=os.getpid(), gpus=gpus,
              started_unix=started, completed_trainings=24, planned_trainings=34))
    # Exercise the newly selected physical device before the ten full trajectories.
    smoke = Trial('cv', METHOD, winner['lr'], winner['C'], seed=FINAL_SEEDS[0],
                  stage='smoke', smoke_phase='final')
    run_queue([smoke], gpus=gpus)
    print('GPU 3 smoke passed; starting seeds 20261111 through 20261120 sequentially', flush=True)
    generate()
    finals = run_queue(jobs, gpus=gpus)
    assert len(finals)==10 and [r['seed'] for r in finals]==list(FINAL_SEEDS)
    assert all(r['physical_gpu']==3 and r['completed_epochs']==5 for r in finals)
    save_json(RESULTS/'final_results.json', finals)
    write_csv(RESULTS/'final_results.csv', finals)
    for stage in ('grid_results.json', 'recheck_results.json'):
        for row in json.loads((RESULTS/stage).read_text()):
            audit_trial(Trial(**{k:row[k] for k in Trial.__dataclass_fields__}))
    unchanged = protected_snapshot()==json.loads((RESULTS/'protected_files_before_gpu3.json').read_text())
    assert unchanged, 'Protected historical/data/cache files changed during the formal queue'
    save_json(RESULTS/'completion.json', dict(status='completed', search=20, recheck=4, final=10, total=34,
              epochs_per_training=5, started_unix=previous['started_unix'], finished_unix=time.time(),
              seconds=time.time()-previous['started_unix'], formal_gpu3_seconds=time.time()-started,
              initial_execution_seconds=previous['seconds'], formal_gpus=gpus,
              protected_files_unchanged=unchanged, code_sha256=code_hashes(), scheduler=scheduler_audit(),
              user_interruption_record='stop_record.json', interrupted_attempts_excluded=True))
    save_json(RESULTS/'run_state.json', dict(status='completed', gpus=gpus, pid=os.getpid(),
              started_unix=started, seconds=time.time()-started, completed_trainings=34))
    save_json(RESULTS/'progress.json', dict(status='completed', completed_trainings=34, planned_trainings=34))
    generate()
    paths=[p for p in sorted(RESULTS.rglob('*')) if p.is_file() and p.name not in
           ('result_manifest.json', 'final_gpu3.log')]
    save_json(RESULTS/'result_manifest.json', dict(code_sha256=code_hashes(),
              runner_sha256=file_hash(Path(__file__)),
              files_sha256={str(p.relative_to(RESULTS)):file_hash(p) for p in paths}))
    print('Completed all ten seeds on physical GPU 3. Report: exp12/results/report.md', flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', default=[3])
    args=parser.parse_args()
    try:
        execute(args.gpus)
    except Exception as error:
        save_json(RESULTS/'failure.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
                  traceback=traceback.format_exc(), time_unix=time.time(), launch='final_gpu3'))
        save_json(RESULTS/'run_state.json', dict(status='failed', pid=os.getpid(), gpus=args.gpus,
                  error=f'{type(error).__name__}: {error}'))
        save_json(RESULTS/'progress.json', dict(status='failed', error=f'{type(error).__name__}: {error}'))
        raise
