"""Independently recompute stage grids, full-trial eligibility and winners."""
from exp4.runtime import ROOT, EXP
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from exp4.config import METHODS
from exp4.search import IID_LRS, MF_LR_MULTIPLIERS, radius_points, lr_points, refinement_points, read_full_trial, winner
from exp4.final_runner import make_trials


def main():
    results = EXP / 'results'
    history = json.loads((results / 'search_history.json').read_text())
    selected = json.loads((results / 'selected_configs.json').read_text())
    assert history['status'] == 'completed' and not history['auto_run_final']
    assert [s['stage'] for s in history['stages']] == list(range(7))
    rows = history['trials']
    assert len(rows) == 47
    groups = {stage: [r for r in rows if r['stage'] == stage] for stage in range(7)}
    assert [len(groups[s]) for s in range(7)] == [1, 9, 5, 9, 9, 5, 9]
    probe = groups[0][0]
    _, _, norms = read_full_trial(probe['spec']['trial'], probe=True)
    u50 = float(np.median(norms))
    assert u50 == history['U50'] == selected['U50']
    s1 = winner(groups[1])
    before3 = winner(groups[1] + groups[2])
    iid = winner(groups[1] + groups[2] + groups[3])
    s4 = winner(groups[4])
    before6 = winner(groups[4] + groups[5])
    grids = {1: radius_points(u50, .001), 2: lr_points(s1['R'], IID_LRS),
             3: refinement_points(before3), 4: radius_points(u50, iid['lr']),
             5: lr_points(s4['R'], [iid['lr'] * m for m in MF_LR_MULTIPLIERS]),
             6: refinement_points(before6)}
    initializations, checkpoints, permutations, augmentation = set(), set(), set(), set()
    for stage, grid in grids.items():
        assert [(r['lr'], r['R']) for r in groups[stage]] == grid
        assert all(r['method'] == METHODS[0 if stage < 4 else 1] for r in groups[stage])
    for record in rows:
        assert json.loads(Path(record['spec_path']).read_text()) == record['spec']
        summary, clip_fraction, _ = read_full_trial(record['spec']['trial'], probe=record['stage'] == 0)
        assert record['final_test_top1'] == summary['final']['test_top1']
        assert record['clip_fraction'] == clip_fraction
        assert record['privacy_calibration'] == summary['calibration']
        folder = Path(record['result_dir'])
        cfg = yaml.safe_load((folder / 'config.yaml').read_text())
        assert cfg['adam_state_dtype'] == cfg['adam_direction_arithmetic_dtype'] == 'float64'
        assert cfg['dtype'] == cfg['update_direction_dtype'] == 'float32'
        assert cfg['test_examples'] == 10000
        assert cfg['local_data_root'] == str(ROOT / 'data') and not cfg['download']
        assert (ROOT / cfg['pretrained_cfg']['checkpoint_path']).resolve().is_relative_to(ROOT / 'cache')
        initializations.add(cfg['initialization_sha256'])
        checkpoints.add(cfg['pretrained_cfg']['checkpoint_sha256'])
        permutations.add((folder / 'train_order.npy').read_bytes())
        augmentation.add(tuple(e['augmentation_trace_sha256'] for e in summary['epochs']))
    assert len(initializations) == len(checkpoints) == len(permutations) == len(augmentation) == 1
    for method, stages in ((METHODS[0], (1, 2, 3)), (METHODS[1], (4, 5, 6))):
        best = winner([r for s in stages for r in groups[s]])
        frozen = selected['methods'][method]
        assert (frozen['lr'], frozen['update_clip_norm'], frozen['selection_top1']) == (
            best['lr'], best['R'], best['final_test_top1'])
        assert frozen['privacy_calibration'] == best['privacy_calibration']
    with (results / 'search_summary.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 47
    finals = make_trials(selected, results / 'final')
    assert len(finals) == 6 and {t['seed'] for t in finals} == {20261011, 20261012, 20261013}
    assert not (results / 'final').exists()
    evidence = dict(status='passed', U50=u50, stage_candidate_counts=[len(groups[s]) for s in range(7)],
                    unique_dp_trials=len({r['trial_id'] for r in rows if r['stage']}),
                    reused_dp_entries=sum(r['reused'] for r in rows if r['stage']),
                    all_trials_full_5_epochs_250_steps=True, smoke_excluded=True,
                    fixed_protocol_and_local_inputs_verified=True, paired_initialization_order_augmentation=True,
                    grids_and_selection_recomputed=True, final_trials_not_started=True)
    (results / 'search_verification.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
