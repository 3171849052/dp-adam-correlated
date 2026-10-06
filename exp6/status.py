"""Read-only status of the staged search."""
from exp6.runtime import EXP
import json
from pathlib import Path


def status():
    folder=EXP / 'results/search'
    summaries=[]; active=[]
    for path in sorted((folder / 'trials').glob('*')):
        if (path / 'summary.json').exists():
            summaries.append(json.loads((path / 'summary.json').read_text()))
        elif (path / 'config.json').exists():
            cfg=json.loads((path / 'config.json').read_text())
            lines=(path / 'steps.jsonl').read_text().splitlines() if (path / 'steps.jsonl').exists() else []
            last=json.loads(lines[-1]) if lines else {}
            active.append(dict(**cfg['trial'],step=last.get('step',0),train_loss=last.get('train_loss'),
                               train_top1=last.get('train_top1'),gpu=cfg['physical_gpu']))
    result=dict(completed=len(summaries),active=active)
    if (folder / 'search_summary.json').exists():
        result['search_summary']=json.loads((folder / 'search_summary.json').read_text())
    if summaries:
        result['best_so_far']={m:max(s['final_test_top1'] for s in summaries if s['trial']['method']==m)
                              for m in sorted({s['trial']['method'] for s in summaries})}
    return result


if __name__ == '__main__':
    print(json.dumps(status(),ensure_ascii=False))
