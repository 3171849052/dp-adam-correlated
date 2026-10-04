"""Run exactly two frozen configurations, each with the three final seeds."""
from exp4.runtime import EXP, output_path
import argparse
import json
from pathlib import Path

from exp4.config import METHODS, load_config
from exp4.launch_batch import launch
from exp4.report import write_final_report


def make_trials(frozen, result_root):
    assert frozen['protocol'] == 'exp4/config.yaml'
    assert set(frozen['methods']) == set(METHODS)
    root = output_path(result_root)
    return [dict(method=method, seed=seed,
                 lr=frozen['methods'][method]['lr'],
                 update_clip_norm=frozen['methods'][method]['update_clip_norm'],
                 result_dir=str(root / method / f'seed_{seed}'))
            for method in METHODS for seed in (20261011, 20261012, 20261013)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--result-root', type=Path, default=EXP / 'results/final')
    args = parser.parse_args()
    load_config()
    frozen = json.loads(args.frozen.read_text())
    trials = make_trials(frozen, args.result_root)
    root = output_path(args.result_root)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'frozen_config.json').write_text(json.dumps(frozen, indent=2))
    (root / 'trials.json').write_text(json.dumps(trials, indent=2))
    launch(trials)
    write_final_report(trials, root)


if __name__ == '__main__':
    main()
