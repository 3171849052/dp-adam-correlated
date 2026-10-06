"""Five specified full-budget stages; deterministic tie breaking and trial reuse."""
from exp5.runtime import EXP, ROOT, output_path
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from exp5.config import FIXED, METHODS, SEARCH_SEED, Trial
from exp5.launch_batch import launch, read_completed
from exp5.train import write_json

IID_LRS = (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3)
SELECTION_RULE = ['max_final_test_top1', 'min_lr', 'max_tau']


def winner(records):
    return max(records, key=lambda r: (r['final_test_top1'], -r['lr'], r['tau']))


def local_points(best, tau_grid):
    index = tau_grid.index(best['tau'])
    taus = tau_grid[max(0, index-1):min(len(tau_grid), index+2)]
    return list(itertools.product(taus, (best['lr']/2, best['lr'], best['lr']*2)))


class StagedSearch:
    def __init__(self, root=EXP / 'results/search', launch_fn=launch):
        self.root = output_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.launch = launch_fn
        self.records, self.stages = {}, []
        self.active_stage = None

    def persist(self, status='running'):
        write_json(self.root / 'search_summary.json', dict(status=status, fixed=FIXED,
                   search_seed=SEARCH_SEED, selection_rule=SELECTION_RULE,
                   stages=self.stages, active_stage=self.active_stage, trials=list(self.records.values()),
                   total_completed_search_trials=len(self.records), full_epochs_per_candidate=5,
                   privacy_scope='Nonprivate probe and hyperparameter selection excluded from per-run guarantee'))

    def record(self, job):
        trial = Trial(**job['trial'])
        summary = read_completed(trial, job['result_dir'])
        self.records[trial.identity] = dict(trial.asdict(), trial_id=trial.identity,
                   final_test_top1=summary['final']['test_top1'],
                   source_trial=job['result_dir'], diagnostics=summary['diagnostics'], privacy=summary['privacy'])
        self.persist()

    def stage(self, number, method, points):
        trials = {t.identity: t for tau, lr in points for t in [Trial(method, SEARCH_SEED, lr, tau)]}
        jobs = [dict(trial=t.asdict(), result_dir=str(self.root / 'trials' / t.identity)) for t in trials.values()]
        reused = sum((Path(j['result_dir']) / 'summary.json').exists() for j in jobs)
        print(json.dumps(dict(event='stage_started', stage=number, method=method,
                             candidates=len(jobs), new_trials=len(jobs)-reused, reused=reused)), flush=True)
        self.active_stage = number
        write_json(self.root / f'stage{number}_summary.json', dict(stage=number, method=method,
                   status='running', trial_ids=list(trials), candidates=len(jobs), reused=reused))
        self.persist()
        self.launch(jobs, on_complete=self.record)
        best = winner([self.records[t] for t in trials])
        result = dict(stage=number, method=method, status='completed', trial_ids=list(trials),
                      candidates=len(jobs), new_trials=len(jobs)-reused, reused=reused, winner=best)
        self.active_stage = None
        self.stages.append(result)
        write_json(self.root / f'stage{number}_summary.json', result)
        self.persist()
        print(json.dumps(dict(event='stage_completed', stage=number, winner=best)), flush=True)
        return best

    def best(self, method):
        return winner([r for r in self.records.values() if r['method'] == method])

    def freeze(self, method, filename):
        selected = dict(self.best(method), selection_rule=SELECTION_RULE, fixed=FIXED, frozen=True)
        path = self.root / filename
        if path.exists():
            assert json.loads(path.read_text()) == selected, f'Frozen winner differs: {path}'
        else:
            write_json(path, selected)
        return selected

    def run(self, probe=None):
        if probe is None:
            probe_path = self.root / 'scale_probe.json'
            if not probe_path.exists():
                subprocess.run([sys.executable, '-B', '-m', 'exp5.probe'], cwd=ROOT,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'), check=True)
            probe = json.loads(probe_path.read_text())
        grid = probe['tau_grid']
        assert grid == sorted(set(grid)) and len(grid) >= 1
        iid, mf = METHODS
        s1 = self.stage(1, iid, [(tau, 1e-3) for tau in grid])
        self.stage(2, iid, [(s1['tau'], lr) for lr in IID_LRS])
        self.stage(3, iid, local_points(self.best(iid), grid))
        iid_frozen = self.freeze(iid, 'iid_frozen.json')
        tau, lr = iid_frozen['tau'], iid_frozen['lr']
        self.stage(4, mf, [(tau, lr*m) for m in (.25, .5, 1., 2., 4.)])
        best = self.best(mf)
        self.stage(5, mf, [(tau, best['lr']*m) for m in (.5, 1., 2.)])
        mf_frozen = self.freeze(mf, 'bandinvmf_frozen.json')
        selected = dict(fixed=FIXED, methods={iid: iid_frozen, mf: mf_frozen},
                        total_completed_search_trials=len(self.records), probe=probe)
        write_json(self.root / 'selected_configs.json', selected)
        self.persist('completed')
        print(json.dumps(dict(event='search_completed', trials=len(self.records))), flush=True)
        return selected


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1,2', choices=['0,1,2'])
    p.add_argument('--run-finals', action='store_true')
    a = p.parse_args()
    StagedSearch().run()
    if a.run_finals:
        from exp5.final_runner import run
        run()
