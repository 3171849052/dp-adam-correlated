"""Read-only live status: python -m exp7.status [--json]."""
import argparse
import csv
import json
from exp7 import BASE
from exp7.report import rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    state_path = BASE / 'results/search_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'status':'platform validation'}
    candidates = rows()
    state['counts'] = dict(new_trials=sum(r['source'] == 'new' for r in candidates),
                           historical_reused=sum(r['source'] == 'historical' for r in candidates),
                           numerical_failures=sum(r['status'] == 'numerical_failure' for r in candidates))
    active = []
    for directory in sorted((BASE / 'results/trials').glob('*')):
        if not directory.is_dir() or (directory / 'summary.json').exists():
            continue
        metrics = directory / 'metrics.csv'
        epochs = list(csv.DictReader(metrics.open())) if metrics.exists() else []
        active.append(dict(result_dir=str(directory.relative_to(BASE)), last_epoch=epochs[-1] if epochs else None,
                           note='incomplete/running; no automatic recovery'))
    payload = dict(state=state, active_or_incomplete=active, candidates=[{k:r.get(k) for k in
                   ('method','lr','C','eps_scale','status','final_test_top1','source','result_dir')} for r in candidates])
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(state, indent=2))
        for r in active:
            print(r)
        for method in sorted({r['method'] for r in candidates}):
            completed = [r for r in candidates if r['method'] == method and r['status'] == 'completed']
            if completed:
                r = max(completed, key=lambda r:r['final_test_top1'])
                print(f"{method}: Top1={r['final_test_top1']:.4f} lr={r['lr']:g} C={r['C']:g} eps_scale={r['eps_scale']}")

if __name__ == '__main__':
    main()
