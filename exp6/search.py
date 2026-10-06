"""Deterministic staged utility tuning, independent C/LR and shared scale epsilon."""
from exp6.runtime import ROOT, EXP, output_path, require_curve
import argparse
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import itertools
import json
from pathlib import Path
from exp6.config import Trial, METHODS, FIXED, SEARCH_SEED, FINAL_SEEDS
from exp6.launch_batch import launch, read_completed
from exp6.audit import digest, environment, diagnostics


def write(path, value):
    path = output_path(path)
    text = json.dumps(value,indent=2,allow_nan=False)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(text)
    temp.replace(path)


def validate_plan(plan):
    assert plan['protocol'] == 'staged-utility-v2'
    assert plan['seed'] == SEARCH_SEED
    assert plan['selection'] == 'final_test_top1_desc_C_asc_lr_asc'
    assert plan['local_factors'] == [.5,1.,2.]
    assert plan['max_boundary_extensions'] in (0,1,2)
    for key in ('clip_norms','learning_rates','geom_eps_values'):
        assert plan[key] == sorted(set(plan[key])) and min(plan[key]) > 0
    Trial(METHODS[0],SEARCH_SEED,plan['initial_lr'],plan['initial_C'],plan['raw_geom_eps'],plan['physical_batch_size'])


def rank(summary):
    return (-summary['final_test_top1'],summary['trial']['C'],summary['trial']['lr'],
            summary['trial']['geom_eps'],summary['trial_id'])


def best(summaries):
    assert summaries, 'No completed candidates'
    return min(summaries,key=rank)


class Search:
    def __init__(self, plan, folder):
        validate_plan(plan)
        self.plan, self.folder = plan, output_path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.provenance = dict(plan=plan,fixed=FIXED,environment=environment(),
            search_source_sha256={name:digest((EXP / name).read_text()) for name in
                                 ('search.py','launch_batch.py','final_runner.py','report.py')})
        path = self.folder / 'provenance.json'
        if path.exists():
            assert json.loads(path.read_text()) == self.provenance, 'Search plan/code/environment conflict'
        else:
            write(path,self.provenance)
        self.summaries = {}
        for path in sorted((self.folder / 'trials').glob('*/summary.json')):
            summary = json.loads(path.read_text())
            job = self.job(Trial(**summary['trial']))
            assert path.parent == Path(job['result_dir'])
            self.summaries[summary['trial_id']] = read_completed(job)
        # Directories without summary are never recovered or overwritten.
        for path in sorted((self.folder / 'trials').glob('*')):
            assert (path / 'summary.json').is_file(), f'Incomplete tuning trial: {path}'
        self.eligible_ids = set()
        self.initial_ids = set(self.summaries)
        self.trained_ids = set()
        self.reused_ids = set()
        self.stages = []
        self.index()

    def job(self, trial):
        return dict(trial=trial.asdict(),result_dir=str(self.folder / 'trials' / trial.identity))

    def trial(self, summary):
        return Trial(**summary['trial'])

    def pool(self, method, geom_eps=None):
        return [s for s in self.summaries.values() if s['trial_id'] in self.eligible_ids and s['trial']['method'] == method and
                (geom_eps is None or s['trial']['geom_eps'] == geom_eps)]

    def winner(self, method, geom_eps=None):
        return best(self.pool(method,geom_eps))

    def index(self):
        rows = [dict(trial_id=s['trial_id'],fingerprint=s['fingerprint'],trial=s['trial'],
            final_test_top1=s['final_test_top1'],final_privacy=s['final_privacy'],
            result_dir=str(self.folder / 'trials' / s['trial_id'])) for s in sorted(self.summaries.values(),key=lambda s:s['trial_id'])]
        write(self.folder / 'completed_trials.json',dict(count=len(rows),trials=rows))

    def stage(self, name, trials, note):
        trials = list({t.identity:t for t in trials}.values())
        jobs = [self.job(t) for t in trials]
        outcome = launch(jobs)
        self.trained_ids.update(outcome['trained'])
        self.reused_ids.update(outcome['reused'])
        for job in jobs:
            s = read_completed(job)
            self.summaries[s['trial_id']] = s
        self.eligible_ids.update(t.identity for t in trials)
        summaries = [self.summaries[t.identity] for t in trials]
        self.index()
        result = dict(stage=name,note=note,selection=self.plan['selection'],
            candidates=[dict(trial=s['trial'],trial_id=s['trial_id'],fingerprint=s['fingerprint'],
                             final_test_top1=s['final_test_top1'],diagnostic_artifacts=str(self.folder / 'trials' / s['trial_id'])) for s in summaries],
            candidate_winner=best(summaries)['trial_id'])
        path = self.folder / (name+'_summary.json')
        if path.exists():
            assert json.loads(path.read_text()) == result, f'Deterministic stage conflict: {path}'
        else:
            write(path,result)
        self.stages.append(name)
        print(json.dumps(dict(event='stage_completed',stage=name,trained=len(outcome['trained']),
            reused=len(outcome['reused']),winner=best(summaries)['trial'],final_test_top1=best(summaries)['final_test_top1'])),flush=True)
        return best(summaries)

    def tune(self, prefix, anchor):
        method, eps = anchor.method, anchor.geom_eps
        self.stage(prefix+'_C', [replace(anchor,C=C) for C in sorted(set(self.plan['clip_norms']+[anchor.C]))],
                   'Coarse C search; LR/geometry fixed at anchor')
        chosen = self.trial(self.winner(method,eps))
        self.stage(prefix+'_lr',[replace(chosen,lr=lr) for lr in sorted(set(self.plan['learning_rates']+[chosen.lr]))],
                   'Coarse LR search at current winning C')
        center = self.trial(self.winner(method,eps))
        self.stage(prefix+'_local',[replace(center,C=center.C*cf,lr=center.lr*lf)
                  for cf,lf in itertools.product(self.plan['local_factors'],repeat=2)],
                   '3x3 local refinement about coarse winner')
        # Only follow an outward local winner. A bracketed direction stops here.
        stops = {}
        for axis in ('C','lr'):
            values = [getattr(center,axis)*f for f in self.plan['local_factors']]
            for attempt in range(self.plan['max_boundary_extensions']):
                current = self.winner(method,eps)
                trial = self.trial(current)
                coordinate = getattr(trial,axis)
                if min(values) < coordinate < max(values):
                    stops[axis] = 'bracketed'
                    break
                direction = .5 if coordinate <= min(values) else 2.
                proposed = replace(trial,**{axis:coordinate*direction})
                outward = self.stage(prefix+f'_{axis}_boundary_{attempt+1}',[proposed],
                                     'One bounded extension of an outward winner')
                values.append(getattr(proposed,axis))
                if outward['final_test_top1'] <= current['final_test_top1']:
                    stops[axis] = 'no_utility_improvement'
                    break
            else:
                stops[axis] = 'extension_budget_reached'
        write(self.folder / (prefix+'_stopping.json'),stops)
        return self.winner(method,eps)

    def freeze(self, prefix, summary):
        trial = self.trial(summary)
        frozen = dict(status='frozen',method=trial.method,fixed=FIXED,search_seed=SEARCH_SEED,
            hyperparameters=dict(C=trial.C,lr=trial.lr,geom_eps=trial.geom_eps),
            geometry_active=trial.scaled,physical_batch_size=trial.physical_batch_size,
            final_test_top1=summary['final_test_top1'],selected_trial_id=trial.identity,
            selected_fingerprint=summary['fingerprint'],selection=self.plan['selection'],
            privacy=summary['calibration'],final_privacy=summary['final_privacy'],
            workload=dict(mechanism=trial.noise,beta1=.9,T=250,num_bands=4 if trial.noise!='iid' else None,
                          implementation='exp2.bandinvmf.build_matrices',matrix_artifact=str(self.folder / 'trials' / trial.identity / 'matrices.npz')),
            diagnostics=diagnostics(self.folder / 'trials' / trial.identity),
            result_dir=str(self.folder / 'trials' / trial.identity))
        path = self.folder / (trial.method+'_frozen.json')
        if path.exists():
            assert json.loads(path.read_text()) == frozen, f'Frozen configuration conflict: {path}'
        else:
            write(path,frozen)
        return frozen

    def run(self):
        p = self.plan
        a = Trial(METHODS[0],SEARCH_SEED,p['initial_lr'],p['initial_C'],p['raw_geom_eps'],p['physical_batch_size'])
        iid = self.freeze('A',self.tune('A_iid',a))
        b = replace(self.trial(self.summaries[iid['selected_trial_id']]),method=METHODS[1])
        mf = self.freeze('B',self.tune('B_mf',b))
        c = replace(a,method=METHODS[2],lr=iid['hyperparameters']['lr'],C=iid['hyperparameters']['C'])
        geometry = self.stage('C_iid_scale_geometry',[replace(c,geom_eps=eps) for eps in p['geom_eps_values']],
                              'Geometry candidates share IID winner C/LR; select only by final_test_top1')
        eps = geometry['trial']['geom_eps']
        write(self.folder / 'scale_geometry_frozen.json',dict(geom_eps=eps,anchor_trial=geometry['trial'],
             selection='final_test_top1; C/lr tie break, then smaller geom_eps',selected_trial_id=geometry['trial_id']))
        scaled = self.freeze('C',self.tune('C_iid_scale',replace(c,geom_eps=eps)))
        d = Trial(METHODS[3],SEARCH_SEED,scaled['hyperparameters']['lr'],scaled['hyperparameters']['C'],eps,p['physical_batch_size'])
        mf_scaled = self.freeze('D',self.tune('D_mf_scale',d))
        frozen = [iid,mf,scaled,mf_scaled]
        selected = dict(status='frozen',fixed=FIXED,search_seed=SEARCH_SEED,final_seeds=list(FINAL_SEEDS),
            selection=p['selection'],shared_scale_geom_eps=eps,methods={f['method']:f for f in frozen},
            provenance_sha256=digest(self.provenance))
        write(self.folder / 'selected_configs.json',selected)
        # Build a reviewable manifest only: no call to final_runner.run/launch.
        from exp6.final_runner import final_jobs
        final = final_jobs(selected,list(FINAL_SEEDS))
        write(self.folder / 'prepared_final_manifest.json',dict(selected_configs=str(self.folder / 'selected_configs.json'),
              seeds=list(FINAL_SEEDS),jobs=final,status='prepared_not_launched'))
        summary = dict(status='completed',selection=p['selection'],plan=p,stages=self.stages,
            total_unique_tuning_trials=len(self.summaries),actual_trained_trials=len(self.summaries)-len(self.initial_ids),
            reused_preexisting_trials=len(self.initial_ids),stage_reuse_events=len(self.reused_ids),
            selected={f['method']:dict(hyperparameters=f['hyperparameters'],final_test_top1=f['final_test_top1'],selected_trial_id=f['selected_trial_id']) for f in frozen},
            final_trials_prepared=len(final),final_launched=False,provenance_sha256=digest(self.provenance))
        # Session counts persist separately so resuming never hides prior work.
        session = dict(started_with_completed=len(self.initial_ids),trained_trial_ids=sorted(self.trained_ids),
                       reused_trial_ids=sorted(self.reused_ids),ended_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        history_path = self.folder / 'sessions.json'
        sessions = json.loads(history_path.read_text()) if history_path.exists() else []
        sessions.append(session); write(history_path,sessions)
        summary['actual_trained_trials_total'] = len(self.summaries)
        write(self.folder / 'search_summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
        return summary


def run(plan_path, folder=EXP / 'results/search'):
    require_curve()
    plan = json.loads(Path(plan_path).read_text())
    with (EXP / 'runtime/search.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        return Search(plan,folder).run()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=EXP / 'search_plan.json')
    p.add_argument('--gpus',choices=['1,2,3'],default='1,2,3')
    p.add_argument('--dry-run',action='store_true')
    a = p.parse_args()
    require_curve()
    if a.dry_run:
        plan = json.loads(a.plan.read_text()); validate_plan(plan)
        print(json.dumps(dict(plan=plan,order=list(METHODS),final_seeds=list(FINAL_SEEDS),final_auto_launch=False)))
    else:
        run(a.plan)


if __name__ == '__main__':
    main()
