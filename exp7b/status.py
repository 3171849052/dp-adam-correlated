"""Read experiment progress without touching training or frozen settings."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

def snapshot():
    live = {}
    scheduler = BASE / 'results/scheduler.jsonl'
    for line in scheduler.read_text().splitlines() if scheduler.exists() else []:
        event = json.loads(line)
        if event['event'] == 'start':
            live[event['gpu']] = event
        else:
            live.pop(event['gpu'], None)
    active = []
    for gpu, event in sorted(live.items()):
        directory = BASE / 'results' / event['category'] / 'trials' / event['trial_id']
        progress = {}
        log = directory / 'train.log'
        if log.exists():
            for line in reversed(log.read_text().splitlines()):
                if line.startswith('{'):
                    row = json.loads(line)
                    if 'progress_step' in row or 'logical_steps' in row:
                        progress = row
                        break
        active.append(dict(gpu=gpu, method=event['method'], seed=event['seed'],
                           lr=event['lr'], C=event['C'], eps_scale=event['eps_scale'],
                           category=event['category'], pid=event['pid'],
                           process_alive=Path(f"/proc/{event['pid']}").exists(),
                           step=progress.get('progress_step', progress.get('logical_steps',0)),
                           epoch=progress.get('epoch',0)))
    counts, winners, final_scores = {}, {}, {}
    for category in ('search','final'):
        rows = [json.loads(p.read_text()) for p in (BASE / f'results/{category}/trials').glob('*/summary.json')]
        counts[category] = {s: sum(r['status']==s for r in rows) for s in ('completed','numerical_failure','failed')}
        for r in rows:
            if r['status'] != 'completed':
                continue
            method = r['method']
            if category == 'search' and (method not in winners or r['final_test_top1']>winners[method]['final_test_top1']):
                winners[method] = {k:r[k] for k in ('lr','C','eps_scale','final_test_top1')}
            elif category == 'final':
                final_scores.setdefault(method,[]).append(r['final_test_top1'])
    summaries = {m:dict(n=len(v),mean=sum(v)/len(v)) for m,v in final_scores.items()}
    pid_path = BASE / 'runtime/stage2.pid'
    driver_pid = int(pid_path.read_text()) if pid_path.exists() else None
    return dict(counts=counts,active=active,search_winners=winners,final_means=summaries,
                driver_alive=driver_pid is not None and Path(f'/proc/{driver_pid}').exists(),
                configs_frozen=(BASE/'results/frozen_configs.json').exists(),
                report_exists=(BASE/'results/final_report.md').exists())

if __name__ == '__main__':
    result = snapshot()
    (BASE / 'results/status.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
