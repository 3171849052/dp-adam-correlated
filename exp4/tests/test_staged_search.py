import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from exp4.config import METHODS, load_config
from exp4.search import StagedSearch, read_full_trial, trial_id


def fake_full_trial(trial):
    folder = Path(trial['result_dir'])
    folder.mkdir(parents=True)
    probe = trial.get('scale_probe', False)
    cfg = copy.deepcopy(load_config())
    cfg['seed'] = trial['seed']
    cfg['optimizer']['lr'] = trial['lr']
    cfg['optimizer']['update_clip_norm'] = None if probe else trial['update_clip_norm']
    cfg.update(effective_epochs=5, effective_total_steps=250, visible_devices='0',
               update_clipping_enabled=not probe, dp_noise_enabled=not probe)
    (folder / 'config.yaml').write_text(yaml.safe_dump(cfg))
    epochs = [dict(logical_steps=50 * epoch, test_top1=.01, epoch_clip_fraction=0. if probe else 1.,
                   gdp_epsilon=None if probe else 8., test_loss=4.6) for epoch in range(1, 6)]
    privacy = dict(privacy_applied=False) if probe else dict(participation='full_temporal',
                                                            per_step_sensitivity=2 * trial['update_clip_norm'])
    summary = dict(status='completed', smoke=False, optimizer_steps=250, physical_batches=1000,
                   noise_steps=0 if probe else 250, seed=trial['seed'], method=trial['method'],
                   lr=trial['lr'], update_clip_norm=None if probe else trial['update_clip_norm'],
                   scale_probe=probe, calibration=privacy, epochs=epochs, final=epochs[-1])
    (folder / 'summary.json').write_text(json.dumps(summary))
    with (folder / 'steps.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'raw_update_norm', 'clip_indicator', 'clip_scale', 'noise_marginal_std'])
        writer.writeheader()
        for step in range(1, 251):
            writer.writerow(dict(step=step, raw_update_norm=512., clip_indicator=int(not probe),
                                 clip_scale=1. if probe else .5, noise_marginal_std=0. if probe else 1.))


def test_all_stages_follow_dependencies_reuse_and_ties_without_final_runs(tmp_path):
    executed = []
    def fake_launch(trials, on_complete=None):
        for trial in trials:
            assert trial['seed'] == 20261001 and not trial.get('smoke', False)
            assert not (Path(trial['result_dir']) / 'summary.json').exists()
            executed.append(trial)
            fake_full_trial(trial)
            if on_complete:
                on_complete(trial)
    search = StagedSearch(tmp_path, launch_fn=fake_launch)
    selected = search.run()
    assert selected['U50'] == 512.
    assert len(executed) == 41
    assert selected['methods'][METHODS[0]]['update_clip_norm'] == 1.
    assert selected['methods'][METHODS[0]]['lr'] == .000125
    assert selected['methods'][METHODS[1]]['update_clip_norm'] == 1.
    assert selected['methods'][METHODS[1]]['lr'] == .000015625
    history = search.history
    assert [s['stage'] for s in history['stages']] == list(range(7))
    assert len(history['trials']) == 47
    assert sum(r['reused'] for r in history['trials']) == 6
    assert all(r['optimizer_steps'] == 250 for r in history['trials'])
    assert history['trials'][0]['participates_in_ranking'] is False
    assert [s['new_trials'] for s in history['stages'][1:]] == [9, 4, 7, 9, 4, 7]
    assert all(r['spec']['config']['optimizer']['beta1'] == .9 for r in history['trials'])
    assert all(r['spec']['config']['bandinvmf_num_bands'] == 4 for r in history['trials'])
    assert not (tmp_path / 'final').exists()
    for name in ('search_summary.csv', 'search_history.json', 'search_report.md', 'selected_configs.json'):
        assert (tmp_path / name).exists()
    assert trial_id(METHODS[0], .001 / 2, 4.) == trial_id(METHODS[0], .0005, 2. * 2)
    assert trial_id(METHODS[0], .001, 4.) != trial_id(METHODS[1], .001, 4.)


def test_smoke_and_incomplete_trials_cannot_be_reused(tmp_path):
    trial = dict(method=METHODS[0], seed=20261001, lr=.001, update_clip_norm=2.,
                 result_dir=str(tmp_path / 'trial'))
    fake_full_trial(trial)
    path = Path(trial['result_dir']) / 'summary.json'
    good = json.loads(path.read_text())
    for update in (dict(smoke=True), dict(optimizer_steps=2), dict(seed=20261011), dict(noise_steps=2)):
        invalid = dict(good, **update)
        path.write_text(json.dumps(invalid))
        with pytest.raises(AssertionError):
            read_full_trial(trial)
    path.write_text(json.dumps(good))
    _, clip_fraction, norms = read_full_trial(trial)
    assert clip_fraction == 1 and np.median(norms) == 512
