"""Auditable Stage 0–6 UC search; stop after freezing the two winners."""
from exp4.runtime import EXP, output_path
import argparse
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from exp4.config import METHODS, load_config
from exp4.launch_batch import launch

FRACTIONS = [2.0 ** -i for i in range(8, -1, -1)]
IID_LRS = [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3]
MF_LR_MULTIPLIERS = [.25, .5, 1., 2., 4.]
LOCAL_MULTIPLIERS = [.5, 1., 2.]
STAGE_NAMES = ['nonprivate_scale_probe', 'iid_radius', 'iid_lr', 'iid_refinement',
               'bandinvmf_radius', 'bandinvmf_lr', 'bandinvmf_refinement']
SELECTION_RULE = ['max_final_test_top1', 'min_update_clip_norm', 'min_lr']


def radius_points(u50, lr):
    return [(lr, u50 * fraction) for fraction in FRACTIONS]


def lr_points(radius, learning_rates):
    return [(lr, radius) for lr in learning_rates]


def refinement_points(best):
    return [(best['lr'] * lm, best['R'] * rm)
            for rm, lm in itertools.product(LOCAL_MULTIPLIERS, repeat=2)]


def winner(records):
    assert records and all(r['status'] == 'completed' and r['stage'] != 0 for r in records)
    return max(records, key=lambda r: (r['final_test_top1'], -r['R'], -r['lr']))


def selection_reason(records, selected):
    unique = {r['trial_id']: r for r in records}
    ties = [r for r in unique.values() if r['final_test_top1'] == selected['final_test_top1']]
    return dict(rule=SELECTION_RULE, eligible_unique_trials=len(unique),
                trials_tied_at_best_top1=len(ties), selected_trial_id=selected['trial_id'],
                final_test_top1=selected['final_test_top1'], R=selected['R'], lr=selected['lr'],
                explanation='Highest final epoch test top1 over all eligible full trials; ties use smaller R, then smaller lr; test loss is not a tie-breaker.')


def trial_id(method, lr, radius):
    key = f'{method}|20261001|{float(lr).hex()}|{float(radius).hex()}'
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def read_full_trial(trial, probe=False):
    folder = Path(trial['result_dir'])
    summary = json.loads((folder / 'summary.json').read_text())
    cfg = yaml.safe_load((folder / 'config.yaml').read_text())
    assert summary['status'] == 'completed' and not summary['smoke']
    assert summary['optimizer_steps'] == 250 and summary['physical_batches'] == 1000
    assert summary['seed'] == cfg['seed'] == 20261001
    assert len(summary['epochs']) == 5
    assert [e['logical_steps'] for e in summary['epochs']] == [50, 100, 150, 200, 250]
    assert cfg['effective_epochs'] == 5 and cfg['effective_total_steps'] == 250
    assert summary['method'] == trial['method'] and summary['lr'] == trial['lr']
    assert cfg['bandinvmf_workload'] == 'sgd' and cfg['bandinvmf_num_bands'] == 4
    assert cfg['visible_devices'] in ('0', '2', '3')
    for key, expected in load_config().items():
        if key not in ('optimizer', 'seed'):
            assert cfg[key] == expected, key
    assert {k: cfg['optimizer'][k] for k in ('beta1', 'beta2', 'eps', 'weight_decay')} == dict(
        beta1=.9, beta2=.999, eps=1e-8, weight_decay=0.0)
    assert summary['scale_probe'] == probe
    assert cfg['update_clipping_enabled'] == cfg['dp_noise_enabled'] == (not probe)
    if probe:
        assert summary['noise_steps'] == 0 and summary['update_clip_norm'] is None
        assert summary['calibration']['privacy_applied'] is False
    else:
        assert summary['noise_steps'] == 250
        assert summary['update_clip_norm'] == trial['update_clip_norm'] == cfg['optimizer']['update_clip_norm']
        assert summary['calibration']['participation'] == 'full_temporal'
        assert summary['calibration']['per_step_sensitivity'] == 2 * trial['update_clip_norm']
        assert np.isclose(summary['final']['gdp_epsilon'], 8.)
    with (folder / 'steps.csv').open() as f:
        steps = list(csv.DictReader(f))
    assert len(steps) == 250 and [int(s['step']) for s in steps] == list(range(1, 251))
    raw_norms = np.array([float(s['raw_update_norm']) for s in steps])
    assert np.isfinite(raw_norms).all()
    if probe:
        assert all(float(s['clip_scale']) == 1 and int(s['clip_indicator']) == 0 and
                   float(s['noise_marginal_std']) == 0 for s in steps)
    return summary, float(np.mean([int(s['clip_indicator']) for s in steps])), raw_norms


class StagedSearch:
    def __init__(self, result_root=EXP / 'results', launch_fn=launch):
        self.root = output_path(result_root)
        self.search_root = self.root / 'search'
        self.spec_root = self.search_root / 'specs'
        self.spec_root.mkdir(parents=True, exist_ok=True)
        self.launch = launch_fn
        self.cfg = load_config()
        self.history = dict(protocol='exp4/config.yaml', search_seed=20261001,
                            status='running', selection_rule=SELECTION_RULE,
                            stages=[], trials=[], auto_run_final=False,
                            interrupted_attempts=[json.loads(p.read_text()) for p in sorted(
                                (self.search_root / 'failed_attempts').glob('*/manifest.json'))],
                            excluded_precision_archives=[json.loads(p.read_text()) for p in sorted(
                                (self.root / 'search_archives').glob('*/manifest.json'))])

    def full_spec(self, trial, stage):
        cfg = copy.deepcopy(self.cfg)
        cfg['seed'] = 20261001
        cfg['optimizer']['lr'] = trial['lr']
        cfg['optimizer']['update_clip_norm'] = None if stage == 0 else trial['update_clip_norm']
        return dict(stage=stage, stage_name=STAGE_NAMES[stage], trial=trial, config=cfg,
                    smoke=False, effective_epochs=5, effective_total_steps=250,
                    update_clipping_enabled=stage != 0, dp_noise_enabled=stage != 0,
                    participates_in_ranking=stage != 0)

    def persist(self):
        (self.root / 'search_history.json').write_text(json.dumps(self.history, indent=2, allow_nan=False))
        completed = [r for r in self.history['trials'] if r['status'] == 'completed']
        if completed:
            fields = ['stage', 'stage_name', 'candidate', 'method', 'trial_id', 'reused',
                      'final_test_top1', 'clip_fraction', 'final_epoch_clip_fraction', 'R',
                      'R_over_U50', 'lr', 'optimizer_steps', 'privacy_calibration', 'spec_path', 'result_dir']
            with (self.root / 'search_summary.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for r in completed:
                    row = {k: r[k] for k in fields}
                    row['privacy_calibration'] = json.dumps(row['privacy_calibration'], sort_keys=True)
                    writer.writerow(row)

    def finish_record(self, record, trial, probe=False):
        summary, clip_fraction, _ = read_full_trial(trial, probe=probe)
        record.update(status='completed', final_test_top1=summary['final']['test_top1'],
                      clip_fraction=clip_fraction,
                      final_epoch_clip_fraction=summary['final']['epoch_clip_fraction'],
                      optimizer_steps=250, privacy_calibration=summary['calibration'],
                      epochs=summary['epochs'])
        self.persist()

    def run_probe(self):
        trial = dict(method=METHODS[0], seed=20261001, lr=1e-3, update_clip_norm=1.,
                     scale_probe=True, result_dir=str(self.search_root / 'stage0_probe'))
        spec_path = self.spec_root / 'stage0_probe.json'
        spec = self.full_spec(trial, 0)
        spec_path.write_text(json.dumps(spec, indent=2))
        reused = (Path(trial['result_dir']) / 'summary.json').exists()
        record = dict(stage=0, stage_name=STAGE_NAMES[0], candidate=0, method='nonprivate-scale-probe',
                      trial_id='stage0_probe', reused=reused, status='planned', R=None,
                      R_over_U50=None, lr=1e-3, spec=spec, spec_path=str(spec_path),
                      result_dir=trial['result_dir'], participates_in_ranking=False)
        self.history['trials'].append(record)
        self.persist()
        if not reused:
            self.launch([trial])
        self.finish_record(record, trial, probe=True)
        _, _, raw_norms = read_full_trial(trial, probe=True)
        self.u50 = float(np.median(raw_norms))
        assert self.u50 > 0
        self.history['U50'] = self.u50
        stage = dict(stage=0, stage_name=STAGE_NAMES[0], status='completed', U50=self.u50,
                     selection_reason='Median of all 250 raw Adam direction norms from full nonprivate probe; excluded from DP method ranking.')
        self.history['stages'].append(stage)
        (self.search_root / 'scale_probe.json').write_text(json.dumps(dict(
            U50=self.u50, raw_update_norms=raw_norms.tolist(), trial=trial, nonprivate=True,
            participates_in_ranking=False), indent=2))
        self.persist()
        print(json.dumps(stage), flush=True)

    def run_stage(self, stage, method, points):
        stage_record = dict(stage=stage, stage_name=STAGE_NAMES[stage], status='running',
                            method=method, candidate_count=len(points))
        self.history['stages'].append(stage_record)
        records, pending, lookup = [], [], {}
        for candidate, (lr, radius) in enumerate(points):
            identity = trial_id(method, lr, radius)
            trial = dict(method=method, seed=20261001, lr=lr, update_clip_norm=radius,
                         result_dir=str(self.search_root / method / identity))
            spec_path = self.spec_root / f'stage{stage}_{candidate:02d}_{identity}.json'
            spec = self.full_spec(trial, stage)
            spec_path.write_text(json.dumps(spec, indent=2))
            reused = (Path(trial['result_dir']) / 'summary.json').exists()
            record = dict(stage=stage, stage_name=STAGE_NAMES[stage], candidate=candidate,
                          method=method, trial_id=identity, reused=reused, status='planned',
                          R=radius, R_over_U50=radius / self.u50, lr=lr,
                          spec=spec, spec_path=str(spec_path), result_dir=trial['result_dir'],
                          participates_in_ranking=True)
            records.append(record)
            self.history['trials'].append(record)
            if reused:
                self.finish_record(record, trial)
            else:
                pending.append(trial)
                lookup[trial['result_dir']] = record
        self.persist()
        print(json.dumps(dict(event='stage_started', **stage_record,
                              new_trials=len(pending), reused_trials=len(records) - len(pending))), flush=True)
        if pending:
            self.launch(pending, on_complete=lambda trial: self.finish_record(lookup[trial['result_dir']], trial))
        assert all(r['status'] == 'completed' for r in records)
        selected = winner(records)
        stage_record.update(status='completed', new_trials=len(pending), reused_trials=len(records) - len(pending),
                            stage_winner=selection_reason(records, selected))
        self.persist()
        print(json.dumps(dict(event='stage_completed', **stage_record)), flush=True)
        return selected

    def eligible(self, method):
        return [r for r in self.history['trials'] if r['method'] == method and r['stage'] != 0]

    def freeze_method(self, method):
        records = self.eligible(method)
        selected = winner(records)
        return dict(lr=selected['lr'], update_clip_norm=selected['R'], R_over_U50=selected['R_over_U50'],
                    search_seed=20261001, source_trial=selected['result_dir'],
                    selection_top1=selected['final_test_top1'], clip_fraction=selected['clip_fraction'],
                    privacy_calibration=selected['privacy_calibration'],
                    selection_reason=selection_reason(records, selected))

    def run(self):
        self.run_probe()
        iid, mf = METHODS
        s1 = self.run_stage(1, iid, radius_points(self.u50, 1e-3))
        self.run_stage(2, iid, lr_points(s1['R'], IID_LRS))
        self.run_stage(3, iid, refinement_points(winner(self.eligible(iid))))
        iid_frozen = self.freeze_method(iid)
        (self.search_root / 'iid_frozen.json').write_text(json.dumps(iid_frozen, indent=2))
        iid_lr = iid_frozen['lr']
        s4 = self.run_stage(4, mf, radius_points(self.u50, iid_lr))
        self.run_stage(5, mf, lr_points(s4['R'], [iid_lr * m for m in MF_LR_MULTIPLIERS]))
        self.run_stage(6, mf, refinement_points(winner(self.eligible(mf))))
        selected = dict(protocol='exp4/config.yaml', provenance='strict_stages_0_to_6',
                        U50=self.u50, search_seed=20261001, selection_rule=SELECTION_RULE,
                        methods={iid: iid_frozen, mf: self.freeze_method(mf)}, auto_run_final=False)
        (self.root / 'selected_configs.json').write_text(json.dumps(selected, indent=2, allow_nan=False))
        self.history.update(status='completed', selected_configs=selected)
        self.persist()
        self.write_report(selected)
        print(json.dumps(dict(event='search_completed', selected_configs=str(self.root / 'selected_configs.json'),
                              auto_run_final=False)), flush=True)
        return selected

    def write_report(self, selected):
        records = self.history['trials']
        lines = ['# exp4 staged UC search', '', f'U50 = {self.u50:.12g}; median of 250 full nonprivate probe steps.',
                 'Probe: seed=20261001, lr=1e-3, no clipping, no noise; excluded from ranking.',
                 'All DP candidates: 5 epochs / 250 steps, seed=20261001, SGD/prefix bandwidth=4.',
                 'Selection: highest final test top1; ties use smaller R, then smaller lr.',
                 'Final seeds were not started.', '',
                 '| stage | candidates | new | reused | stage winner top1 | R | lr |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for stage in self.history['stages'][1:]:
            best = stage['stage_winner']
            lines.append(f"| {stage['stage']} {stage['stage_name']} | {stage['candidate_count']} | "
                         f"{stage['new_trials']} | {stage['reused_trials']} | {best['final_test_top1']:.4f} | "
                         f"{best['R']:.12g} | {best['lr']:.12g} |")
        lines += ['', '| frozen method | final test top1 | R | R/U50 | lr | clip fraction |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for method, cfg in selected['methods'].items():
            lines.append(f"| {method} | {cfg['selection_top1']:.4f} | {cfg['update_clip_norm']:.12g} | "
                         f"{cfg['R_over_U50']:.12g} | {cfg['lr']:.12g} | {cfg['clip_fraction']:.4f} |")
            lines += ['', f"{method} selection reason: {json.dumps(cfg['selection_reason'], ensure_ascii=False)}", '']
        lines += [f"Unique DP trials: {len({r['trial_id'] for r in records if r['stage'] != 0})}.",
                  f"Reused candidate entries: {sum(r['reused'] for r in records if r['stage'] != 0)}.",
                  'Raw diagnostics and the nonprivate probe are research artifacts; epsilon=8 is per DP trial, not a total search budget.',
                  'Test-based tuning reuses the test set; final multiseed results are not an independent holdout estimate.', '']
        (self.root / 'search_report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    StagedSearch().run()


if __name__ == '__main__':
    main()
