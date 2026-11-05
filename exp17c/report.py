"""CSV, paired Student-t intervals and Chinese report from completed trials."""
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exp17c.common import RESULTS,read

def report():
    summaries=[];history=[]
    for path in sorted((RESULTS/'trials').glob('*/summary.json')):
        s=read(path)
        if s['status']!='completed' or s['config']['stage']=='smoke':continue
        cfg=s['config']
        summaries.append(dict(**cfg,loss=s['validation_loss'],accuracy=s['accuracy'],epsilon=s['actual_epsilon'],delta=1e-5))
        evals={r['step']:r for r in read(path.parent/'evaluations.json')}
        for line in (path.parent/'diagnostics.jsonl').read_text().splitlines():
            r=json.loads(line);r.update(evals.get(r['step'],{}));history.append(dict(**cfg,**r))
    df=pd.DataFrame(summaries);h=pd.DataFrame(history);pairs=[];intervals=[]
    if len(df):
        df.to_csv(RESULTS/'results.csv',index=False);h.to_csv(RESULTS/'diagnostics.csv',index=False)
        for stage,group in df.groupby('stage'):
            baseline=group[group.alpha==0].set_index('seed')
            for a,g in group[group.alpha>0].groupby('alpha'):
                joined=g.set_index('seed').join(baseline,rsuffix='_baseline',how='inner')
                for metric in ['loss','accuracy']:
                    diff=joined[metric]-joined[metric+'_baseline']
                    for seed,value in diff.items():pairs.append(dict(stage=stage,alpha=a,seed=seed,metric=metric,difference=value))
                    n=len(diff)
                    if n<2:continue
                    mean=diff.mean();half=t.ppf(.975,n-1)*diff.std(ddof=1)/np.sqrt(n)
                    intervals.append(dict(stage=stage,alpha=a,metric=metric,n=n,mean=mean,low=mean-half,high=mean+half))
        pd.DataFrame(pairs).to_csv(RESULTS/'paired_differences.csv',index=False)
        pd.DataFrame(intervals).to_csv(RESULTS/'paired_ci95.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    if len(h):
        for (stage,a),g in h.groupby(['stage','alpha']):
            means=g.groupby('step').mean(numeric_only=True)
            for ax,key in zip(axes.flat,['train_loss','validation_loss','accuracy','correction_norm','relative_update_scale','correction_base_cosine']):
                ax.plot(means.index,means[key],label=f'{stage} a={a:g}');ax.set_title(key);ax.set_xlabel('step')
        axes[0,0].legend(fontsize=6)
    fig.tight_layout();fig.savefig(RESULTS/'training_curves.png');plt.close(fig)
    lines=['# Exp17c：r=0 的 DP-Adam-LC α 搜索','',
        '固定 lr=0.001187378144539193、eps=1e-8；无额外分母平滑、曲率裁剪或截断。修正仅使用裁剪聚合后加高斯噪声的梯度。Adam 历史状态为 FP32，解析曲率以 FP64 计算，第一步使用精确分段公式，零点修正为零。',
        '物理 GPU 1、2、3 独立调度。B 为 10×3×50，C 为三个非零 α 加基线、5 个新种子×225，D 为冻结 α 加基线、10 个新种子×250。按末步验证交叉熵均值选择，准确率为次指标。',
        'B/C 使用 225 步噪声校准，D 使用 250 步校准。每次训练独立报告 ε、δ=1e-5；多次调参及重复实验不具有联合 ε=8 保证。验证集视为公开；原始损失、诊断和检查点为内部审计数据。正式测试集只在 α 冻结后用于 D，且历史实验已有测试集暴露。',
        '配对差定义为 LC 减基线；交叉熵负值较好，准确率正值较好。95% 区间为配对种子差的 Student-t 区间，未作多重比较校正；选择阶段区间仅作描述。',
        '状态：'+('D 已完成。' if (RESULTS/'D_results.json').exists() else '实验未完成，不能作正式效果结论。'),
        f'已完成训练数：{len(df)}。']
    if len(df):
        lines+=['','## 结果',df.to_markdown(index=False),'','## 配对 95% 置信区间',pd.DataFrame(intervals).to_markdown(index=False)]
        lines+=['','## 稳定性与修正幅度',
            '第一步的极小带噪输入可能产生较大曲率；应结合 correction_max_abs、各分位数、修正范数与基础范数之比以及方向余弦判断，而不能只看平均准确率。',
            h.groupby(['stage','alpha'])[['correction_norm','correction_max_abs','relative_update_scale','input_min_abs','input_zero_count']].agg(['mean','max']).to_markdown(),
            '修正尺度随 α 直接变化，但后续历史与训练轨迹也会随 α 改变。若交叉熵配对区间跨零，当前种子数量不足以支持明确改进结论。']
    if (RESULTS/'boundary_check.json').exists():lines+=['','边界检查：'+json.dumps(read(RESULTS/'boundary_check.json'),ensure_ascii=False)]
    (RESULTS/'report.md').write_text('\n\n'.join(lines)+'\n')

if __name__=='__main__':report()
