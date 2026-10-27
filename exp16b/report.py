"""Live complete candidate listings and provisional single-seed rankings."""
from collections import Counter
from exp16b import RESULTS
from exp16b.config import METHODS,save_json
from exp16b.support import read,write_csv

def update_report(jobs,status):
    from exp16b.search import GRIDS
    counts=dict(Counter(r['status'] for r in status.values()))
    save_json(RESULTS/'progress.json',dict(total=len(jobs),counts=counts))
    text=['# exp16b 粗搜索进度与分析',str(counts),'仅单种子内部验证；不冻结正式最优配置，不生成细网格、复核或正式训练。']
    winners={}
    for task in ('cv','nlp'):
        for method in METHODS:
            candidates=[]
            for c in jobs:
                if (c.task,c.method)!=(task,method):continue
                r=read(c.output/'summary.json') if (c.output/'summary.json').exists() else {}
                candidates.append(dict(trial_id=c.id,lr=c.lr,clip=c.clip,eps_scale=c.eps_scale,status=status[c.id]['status'],accuracy=r.get('accuracy'),validation_loss=r.get('validation_loss')))
            valid=sorted([r for r in candidates if r['status']=='completed'],key=lambda r:(-r['accuracy'],r['validation_loss'],r['trial_id']))
            ordered=valid+sorted([r for r in candidates if r['status']!='completed'],key=lambda r:r['trial_id'])
            write_csv(RESULTS/f'ranking_{task}_{method}.csv',[dict(rank=i+1 if r in valid else None,**r) for i,r in enumerate(ordered)])
            if valid:
                best=valid[0];boundary={k:('lower' if best[k]==min(v) else 'upper' if best[k]==max(v) else 'interior') for k,v in GRIDS[task].items()}
                winners[f'{task}/{method}']=dict(candidate=best,boundary=boundary,provisional=True)
                text.append(f"{task}/{method}: {len(valid)}/27 完成；当前候选 {best['trial_id']}；边界 {boundary}。")
    save_json(RESULTS/'best_candidates.json',winners)
    (RESULTS/'coarse_report.md').write_text('\n\n'.join(text)+'\n')
