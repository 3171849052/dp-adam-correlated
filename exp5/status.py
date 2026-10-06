"""Read-only progress inspection; no numerical libraries or GPU allocation."""
from exp5.runtime import EXP
import csv
import json


def status():
    search = EXP / 'results/search/search_summary.json'
    if search.exists():
        summary = json.loads(search.read_text())
        print(json.dumps(dict(search_status=summary['status'], stage=summary['active_stage'],
                              completed_search_trials=summary['total_completed_search_trials'])))
        for stage in summary['stages']:
            winner = stage['winner']
            print(json.dumps(dict(stage_winner=stage['stage'], tau=winner['tau'],
                                  lr=winner['lr'], top1=winner['final_test_top1'])))
    for category in ('search/trials', 'final'):
        root = EXP / 'results' / category
        completed = list(root.glob('*/summary.json'))
        if category == 'final' and root.exists():
            print(json.dumps(dict(completed_final_runs=len(completed))))
        for path in sorted(root.glob('*/steps.csv')):
            if (path.parent / 'summary.json').exists():
                continue
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                row = rows[-1]
                print(json.dumps({k: row[k] for k in ('method', 'seed', 'tau', 'lr', 'step')}))
    final = EXP / 'results/final/final_summary.json'
    if final.exists():
        print(json.dumps(json.loads(final.read_text())))

if __name__ == '__main__':
    status()
