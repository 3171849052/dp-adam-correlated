"""Read-only concise status for the staged search."""
from exp4.runtime import EXP
import csv
import json
from pathlib import Path


def main():
    history = json.loads((EXP / 'results/search_history.json').read_text())
    stage = history['stages'][-1] if history['stages'] else history['trials'][0]
    completed = {r['trial_id']: r for r in history['trials'] if r['status'] == 'completed' and r['stage'] != 0}
    active = []
    queued = 0
    for r in history['trials']:
        if r['status'] == 'completed':
            continue
        folder = Path(r['result_dir'])
        if not folder.exists():
            queued += 1
            continue
        steps = 0
        if (folder / 'steps.csv').exists():
            with (folder / 'steps.csv').open() as f:
                steps = sum(1 for _ in csv.DictReader(f))
        epochs = []
        if (folder / 'metrics.csv').exists():
            with (folder / 'metrics.csv').open() as f:
                epochs = list(csv.DictReader(f))
        active.append(dict(R=r['R'], lr=r['lr'], steps=steps,
                           last_epoch_top1=float(epochs[-1]['test_top1']) if epochs else None))
    print(json.dumps(dict(status=history['status'], stage=stage['stage'],
                          stage_name=stage['stage_name'], completed_unique_dp_trials=len(completed),
                          queued=queued, active=active,
                          frozen_iid=(EXP / 'results/search/iid_frozen.json').exists(),
                          selected_configs=(EXP / 'results/selected_configs.json').exists())))


if __name__ == '__main__':
    main()
