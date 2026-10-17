"""Descriptive comparisons only: one paired seed, no significance inference."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exp14a import RESULTS
from exp14a.support import jobs, reference_dir, write_csv

def main():
    rows=[]
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for cfg in jobs():
        path=cfg.output/'summary.json'
        s=json.loads(path.read_text()) if path.exists() else {'status':'missing'}
        baseline=json.loads((reference_dir(cfg)/'summary.json').read_text())
        row=dict(task=cfg.task,method=cfg.method,seed=cfg.seed,lr=cfg.lr,C=cfg.C,status=s['status'],exp9_accuracy=baseline['accuracy'],adam3_accuracy=s.get('accuracy'),difference_pp=100*(s['accuracy']-baseline['accuracy']) if 'accuracy' in s else None,train_seconds=s.get('train_seconds'),error=s.get('error'))
        rows.append(row)
        if s['status']=='completed':
            for key in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace','strategy_sha256','workload_sha256'):
                assert s[key]==baseline[key],(cfg.task,cfg.method,key)
            ax=axes[0 if cfg.task=='cv' else 1]
            label=cfg.method.removeprefix('dp-adam-').removesuffix('-bandinvmf')
            ax[0].plot([e['epoch'] for e in s['epochs']],[e['accuracy'] for e in s['epochs']],label=label+' AdaM3')
            ax[0].plot([e['epoch'] for e in baseline['epochs']],[e['accuracy'] for e in baseline['epochs']],ls='--',label=label+' Exp9')
            ax[1].plot([e['epoch'] for e in s['epochs']],[e['train_loss'] for e in s['epochs']],label=label)
    write_csv(RESULTS/'comparison.csv',rows);write_csv(RESULTS/'summary.csv',rows)
    for i,task in enumerate(('CV','NLP')):
        for j,metric in enumerate(('accuracy','train loss')):
            axes[i,j].set(title=f'{task}: {metric}',xlabel='Epoch');axes[i,j].legend(fontsize=7)
    fig.tight_layout();fig.savefig(RESULTS/'training_curves.png');plt.close(fig)
    lines=['# Exp14a: MF + AdaM³ 对照 Exp9','', '固定种子 20261111；ε=8，δ=1e-5；冻结学习率和裁剪阈值。仅将二阶矩输入改为未偏差修正的一阶矩平方。', '', '|任务|机制|状态|Exp9 准确率|AdaM³ 准确率|差值（百分点）|','|---|---|---|---:|---:|---:|']
    for r in rows:
        val='—' if r['adam3_accuracy'] is None else f"{r['adam3_accuracy']:.4%}"
        diff='—' if r['difference_pp'] is None else f"{r['difference_pp']:+.3f}"
        lines.append(f"|{r['task']}|{r['method']}|{r['status']}|{r['exp9_accuracy']:.4%}|{val}|{diff}|")
    lines+=['','单种子结果仅用于描述性配对比较，不做统计显著性推断。历史评估集并非新的盲测集。','', '所有已完成训练均核对初始化、数据顺序、增强/Dropout 随机轨迹和矩阵哈希与 Exp9 相同。训练耗时包括诊断开销。','', '详见 summary.csv、comparison.csv、training_curves.png 和各任务目录。']
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
