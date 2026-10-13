"""Platform, mathematical tests and real two-step GPU smokes, then stop."""
import argparse
import json
import subprocess
import sys
import traceback
from exp9b import ROOT, RESULTS
from exp9b.config import Trial, METHODS, COUNTS, file_hash, save_json
from exp9b.audit import code_hashes, protected_snapshot, audit_pairing, audit_fifo, full_audit

def unit_tests(prefix=''):
    log_path=RESULTS/f'{prefix}unit_tests.log';xml_path=RESULTS/f'{prefix}tests.xml'
    with log_path.open('w') as log:
        subprocess.run([sys.executable,'-B','-m','pytest','exp9b/tests','-q',
                        '-o','cache_dir=exp9b/runtime/pytest_cache','--basetemp=exp9b/runtime/tests',
                        f'--junitxml={xml_path}'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def smoke_jobs():
    jobs=[]
    for task in ('cv','nlp'):
        phases=('search','final') if task=='cv' else ('search',)
        for phase in phases:
            for method in METHODS:
                jobs.append(Trial(task,method,.001,100 if method.endswith('-scale') else (10 if task=='cv' else 1),
                                  .1 if method.endswith('-scale') else None,stage='smoke',smoke_phase=phase))
    # Eighth NLP smoke fills all three GPUs with two independent workers each.
    jobs.append(Trial('nlp',METHODS[-1],.001,100,.1,seed=20261102,stage='smoke'))
    return jobs

def verify_platform(gpus=(0,1,2)):
    from exp9b.data import prepare
    from exp9b.grid import write_grid
    from exp9b.launcher import run_queue
    from exp9b.report import write_csv,trial_table,compute_report
    RESULTS.mkdir(parents=True,exist_ok=True)
    before=protected_snapshot()
    save_json(RESULTS/'protected_files_before.json',before)
    save_json(RESULTS/'platform_validation.json',dict(status='running',full_experiments_started=False))
    try:
        prepare();write_grid()
        from exp9b.bandmf import prepare_matrices
        prepare_matrices()
        from exp9b.comparison import baseline_evidence, matrix_geometry
        baseline_evidence()
        save_json(RESULTS/'matrix_geometry_validation.json',matrix_geometry())
        unit_tests()
        # Validate the exact requested Stage2 CLI in a non-training mode.
        plan=subprocess.check_output([sys.executable,'-B','-m','exp9b.stage2','--gpus','0','1','2',
                                     '--cv-per-gpu','1','--nlp-per-gpu','2','--plan'],cwd=ROOT,text=True)
        assert json.loads(plan)['counts']==COUNTS
        (RESULTS/'stage2_cli_plan.json').write_text(plan)
        jobs=smoke_jobs()
        # The separate smoke waves directly test two concurrent NLP processes
        # on every GPU, after all CV workers release their slots.
        rows=run_queue([j for j in jobs if j.task=='cv'],gpus=gpus)
        nlp_jobs=[j for j in jobs if j.task=='nlp']
        # First establish single-worker memory feasibility, then test two
        # simultaneous full-batch NLP workers on each assigned GPU.
        rows+=run_queue(nlp_jobs[:1],gpus=gpus,nlp_per_gpu=1)
        rows+=run_queue(nlp_jobs[1:],gpus=gpus,nlp_per_gpu=2)
        assert len(rows)==22 and all(r['status']=='completed' for r in rows)
        pairing=audit_pairing(rows);fifo=audit_fifo()
        assert set(fifo['gpu_starts'])>=set(gpus) or set(map(int,fifo['gpu_starts']))>=set(gpus)
        assert all(fifo['peak_concurrency'][gpu]['nlp']==2 for gpu in gpus)
        assert before==protected_snapshot(), 'Historical experiments, data or cache were modified'
        save_json(RESULTS/'smoke_summary.json',rows);write_csv(RESULTS/'smoke_summary.csv',trial_table(rows))
        compute_report(rows);audit=full_audit()
        evidence=('unit_tests.log','tests.xml','smoke_summary.json','assets_manifest.json',
                  'stage2_cli_plan.json','cv_grid.json','nlp_grid.json','matrix_validation.json','matrix_geometry_validation.json','exp9_comparison_sources.json')
        save_json(RESULTS/'platform_validation.json',dict(status='passed',full_experiments_started=False,
                  code_sha256=code_hashes(),evidence_sha256={n:file_hash(RESULTS/n) for n in evidence},
                  matrix_cache_sha256={str(p.relative_to(RESULTS)):file_hash(p) for p in sorted((RESULTS/'matrix_cache').glob('*'))},
                  smoke_trials=22,smoke_logical_steps_each=2,smoke_horizons=[225,250,310],gpus=list(gpus),
                  pairing=pairing,fifo=fifo,historical_files_unchanged=True,
                  tests='passed',stage2_cli='validated with --plan; no full training started',planned_full_trials=COUNTS))
        lines=['# Exp9b Stage 1','','Platform tests and 22 real GPU smokes passed; Stage 2 has not started.',
               'CV: seven methods at T=225 and T=250, two logical steps each (250 × 4). NLP: seven methods '
               'plus one additional concurrency smoke, two 1000-example steps each, max_length=128.',
               'Two steps exercise previous completed vhat. Matrix/noise/accounting tests cover complete T=225/250/310.',
               'Smoke is a feasibility and numerical check; it does not establish five-epoch accuracy or stability.',
               '', '| Task | Horizon | Method | GPU | Peak allocated GiB | Peak reserved GiB | Seconds/step |',
               '|---|---:|---|---:|---:|---:|---:|']
        lines += [f"| {r['task']} | {r['planned_total_steps']} | {r['method']} | {r['physical_gpu']} | "
                  f"{r['peak_allocated_bytes']/2**30:.3f} | {r['peak_reserved_bytes']/2**30:.3f} | {r['logical_step_seconds']:.3f} |" for r in rows]
        lines += ['', 'Expected full trainings: 321 grid + 56 recheck + 140 formal + 126 fixed-hyperparameter privacy scan = 643.',
                  'Per-run epsilon is verified individually; internal validation is assumed public/non-protected. '
                  'The complete model-selection pipeline is not claimed to be epsilon=8 DP.',
                  'Historical CV test-based selection and previously viewed NLP official validation prevent a new blind-test claim.',
                  '', 'From the repository root:', '', '```bash',
                  'conda run --no-capture-output -n curve python -B -m exp9b.stage2 --gpus 0 1 2 --cv-per-gpu 1 --nlp-per-gpu 2', '```']
        (RESULTS/'stage1_report.md').write_text('\n'.join(lines)+'\n')
        (RESULTS/'final_report.md').write_text('# Exp9b status\n\nStage 1 passed. No full experiment results exist yet. '
                                             'See [Stage 1 validation](stage1_report.md). Stage 2 will generate the paper report.\n')
        print('\n'.join(lines),flush=True)
    except Exception as error:
        save_json(RESULTS/'platform_validation.json',dict(status='failed',error=str(error),traceback=traceback.format_exc(),
                  full_experiments_started=False,physical_batch_policy='No automatic reduction; inspect failures before continuing'))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2],choices=range(4))
    verify_platform(p.parse_args().gpus)
