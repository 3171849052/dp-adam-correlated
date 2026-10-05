"""Deterministic Stage 1–6 search; freeze winners without starting finals."""
from exp5.runtime import EXP, output_path
import itertools
import json
from pathlib import Path
from exp5.config import FIXED, METHODS, SEARCH_SEED, Trial
from exp5.launch_batch import launch, read_completed

CLIPS = (.25, .5, 1., 2., 4.)
IID_LRS = (1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3)
LOCAL = (.5, 1., 2.)
SELECTION_RULE = ['max_final_test_top1', 'min_C', 'min_lr']


def write_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace makes completed stage snapshots safe to resume.
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def winner(records):
    return max(records, key=lambda r: (r['final_test_top1'], -r['C'], -r['lr']))


def local_points(best):
    return [(best['C'] * cm, best['lr'] * lm) for cm, lm in itertools.product(LOCAL, repeat=2)]


class StagedSearch:
    def __init__(self, root=EXP / 'results/search', launch_fn=launch):
        self.root = output_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.launch = launch_fn
        self.records, self.stages = {}, []

    def persist(self, status='running'):
        write_json(self.root / 'search_summary.json', dict(status=status, fixed=FIXED,
                   search_seed=SEARCH_SEED, selection_rule=SELECTION_RULE,
                   stages=self.stages, trials=list(self.records.values()),
                   total_completed_search_trials=len(self.records), auto_run_final=False))

    def record(self, job):
        trial = Trial(**job['trial'])
        summary = read_completed(trial, job['result_dir'])
        self.records[trial.identity] = dict(trial.asdict(), trial_id=trial.identity,
                   final_test_top1=summary['final']['test_top1'],
                   source_trial=job['result_dir'], diagnostics=summary['diagnostics'],
                   privacy=summary['privacy'])
        self.persist()

    def stage(self, number, method, points):
        trials = {Trial(method, SEARCH_SEED, lr, C).identity: Trial(method, SEARCH_SEED, lr, C)
                  for C, lr in points}
        jobs = [dict(trial=t.asdict(), result_dir=str(self.root / 'trials' / t.identity))
                for t in trials.values()]
        reused = sum((Path(j['result_dir']) / 'summary.json').exists() for j in jobs)
        print(json.dumps(dict(event='stage_started', stage=number, method=method,
                              candidates=len(jobs), new_trials=len(jobs) - reused, reused=reused)), flush=True)
        self.launch(jobs, on_complete=self.record)
        best = winner([self.records[t] for t in trials])
        result = dict(stage=number, method=method, status='completed',
                      trial_ids=list(trials), candidates=len(jobs), new_trials=len(jobs) - reused,
                      reused=reused, winner=best)
        self.stages.append(result)
        write_json(self.root / f'stage{number}_summary.json', result)
        self.persist()
        print(json.dumps(dict(event='stage_completed', **result)), flush=True)
        return best

    def best(self, method):
        return winner([r for r in self.records.values() if r['method'] == method])

    def freeze(self, method, filename):
        selected = dict(self.best(method), selection_rule=SELECTION_RULE, fixed=FIXED, frozen=True)
        path = self.root / filename
        if path.exists():
            existing = json.loads(path.read_text())
            assert existing == selected, f'Frozen winner differs: {path}'
        else:
            write_json(path, selected)
        return selected

    def run(self):
        iid, mf = METHODS
        s1 = self.stage(1, iid, [(C, 5e-4) for C in CLIPS])
        self.stage(2, iid, [(s1['C'], lr) for lr in IID_LRS])
        self.stage(3, iid, local_points(self.best(iid)))
        iid_frozen = self.freeze(iid, 'iid_frozen.json')
        iid_lr = iid_frozen['lr']
        s4 = self.stage(4, mf, [(C, iid_lr) for C in CLIPS])
        self.stage(5, mf, [(s4['C'], iid_lr * m) for m in (.25, .5, 1., 2., 4.)])
        self.stage(6, mf, local_points(self.best(mf)))
        mf_frozen = self.freeze(mf, 'bandinvmf_frozen.json')
        selected = dict(fixed=FIXED, methods={iid: iid_frozen, mf: mf_frozen},
                        total_completed_search_trials=len(self.records), auto_run_final=False)
        write_json(self.root / 'selected_configs.json', selected)
        self.persist(status='completed')
        print(json.dumps(dict(event='search_completed', **selected)), flush=True)
        return selected


if __name__ == '__main__':
    StagedSearch().run()
