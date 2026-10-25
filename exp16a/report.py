"""Actual ten-seed results, paired Student-t intervals and separate audit tables."""
from itertools import combinations
import numpy as np
from scipy.stats import t
from exp16a import RESULTS
from exp16a.config import METHODS,FINAL_SEEDS,Trial,file_hash,save_json
from exp16a.support import read,write_csv,verify_protected,code_hashes
from exp16a.scheduler import verify

def stats(values):
    x=np.asarray(values,dtype=float);mean=float(x.mean());sd=float(x.std(ddof=1));half=float(t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)))
    return dict(mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half)
def report():
    from exp16a.stage1 import verify_acceptance
    verify_acceptance();formal=read(RESULTS/'formal_results.json');freeze=read(RESULTS/'final_freeze.json')
    if len(formal)!=280 or any(r['status']!='completed' for r in formal):raise ValueError('Formal evidence incomplete')
    for row in formal:verify(Trial(**{k:row[k] for k in Trial.__dataclass_fields__}))
    summary=[];runtime=[];pairs=[];utility=[];audits={}
    from exp16a.matrices import load,audit_arrays
    for task in ('cv','nlp'):
        groups={}
        for m in METHODS:
            runs=sorted([r for r in formal if r['task']==task and r['method']==m],key=lambda r:r['seed'])
            if [r['seed'] for r in runs]!=list(FINAL_SEEDS):raise ValueError('Missing/duplicate formal seed')
            groups[m]=runs;cfg=freeze['winners'][task][m]
            base=dict(task=task,method=m,lr=cfg['lr'],clip=cfg['clip'],eps_scale=cfg['eps_scale'],n=10)
            row=dict(base)
            for metric,factor in [('accuracy',100),('validation_loss',1),('train_seconds',1),('seconds',1),('peak_allocated_bytes',1),('peak_reserved_bytes',1),('cpu_peak_rss_bytes',1)]:
                for key,value in stats([r[metric]*factor for r in runs]).items():row[f'{metric}_{key}']=value
            row['train_loss_mean']=float(np.mean([r['epochs'][-1]['train_loss'] for r in runs]))
            row['clip_fraction_mean']=float(np.mean([r['epochs'][-1]['clip_fraction'] for r in runs])) if runs[0]['clip'] is not None else None
            summary.append(row)
            runtime.append({k:v for k,v in row.items() if k in base or any(s in k for s in ('seconds','bytes'))})
            for r in runs:utility.append(dict(task=task,method=m,seed=r['seed'],accuracy_percent=r['accuracy']*100,official_loss=r['validation_loss'],train_loss=r['epochs'][-1]['train_loss'],epsilon=r['actual_epsilon'],trial_id=r['trial_id']))
            if runs[0]['clip'] is not None:
                cfgobj=Trial(**{k:runs[0][k] for k in Trial.__dataclass_fields__});a=load(cfgobj.workload,cfgobj.mechanism,cfgobj.total_steps)
                audits[str(a['path'])]=audit_arrays(a['A'],a['C'],a['D'])
        for left,right in combinations(METHODS,2):
            a,b=groups[left],groups[right]
            for ra,rb in zip(a,b):
                for key in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace'):
                    if ra[key]!=rb[key]:raise ValueError(f'Unpaired formal {task}/{left}/{right}/{key}')
            for metric,factor in [('accuracy',100),('validation_loss',1)]:
                pairs.append(dict(task=task,left=left,right=right,metric=metric,units='percentage_points' if metric=='accuracy' else 'cross_entropy',**stats([(rb[metric]-ra[metric])*factor for ra,rb in zip(a,b)])))
    write_csv(RESULTS/'utility.csv',utility);write_csv(RESULTS/'summary.csv',summary);write_csv(RESULTS/'runtime_memory.csv',runtime);write_csv(RESULTS/'paired_differences.csv',pairs)
    # Matrix errors/sensitivity are data-independent diagnostics, not Adam trajectory errors.
    all_matrices=read(RESULTS/'matrix_audit.json')
    write_csv(RESULTS/'matrix_errors.csv',[{k:r[k] for k in ('workload','mechanism','total_steps','objective','mean_workload_error','workload_noise_MSE_clip1','inverse_max_abs','factorization_max_abs')} for r in all_matrices])
    write_csv(RESULTS/'sensitivity.csv',[{k:r[k] for k in ('workload','mechanism','total_steps','sensitivity','absolute_column_upper_bound','aligned_lower_bound','sigma_sum_clip1','epsilon')} for r in all_matrices])
    stage_audit={}
    for stage in ('coarse','fine','review','formal'):
        rows=read(RESULTS/f'{stage}_results.json')
        states=read(RESULTS/f'{stage}_status.json')
        stage_audit[stage]=dict(planned=len(rows),reused=sum(r.get('reused',False) for r in states.values()),completed=sum(r['status']=='completed' for r in rows),excluded=[dict(trial_id=r['trial_id'],reason=r.get('error'),path=str(Trial(**{k:r[k] for k in Trial.__dataclass_fields__}).output)) for r in rows if r['status']!='completed'])
    verify_protected()
    save_json(RESULTS/'final_audit.json',dict(status='passed',code_sha256=code_hashes(),stages=stage_audit,formal=280,configurations=28,matrix_audits=audits,final_freeze_sha256=file_hash(RESULTS/'final_freeze.json'),read_only_snapshot_verified=True))
    lines=['# Exp16a 实测报告','',
        '全部 280 次正式五轮训练完成并通过审计。单次 DP 训练 ε=8、δ=1e-5；add/remove zero-out、固定五轮参与、不使用采样放大。无噪声 Adam 为非私有基线。调参、复核和重复训练未声明组合 ε=8；内部验证假定公开，原始诊断仅作私有审计。',
        '', '选参仅用内部验证末轮准确率和损失。官方评价历史上已经使用过，本次不声称新的盲测。标准 Adam β=(.9,.999)、eps=1e-8；STP 使用 Exp9 的撤销缩放再 Adam 更新。',
        '', '|任务|方法|LR|clip|eps_scale|官方准确率均值 ± SD (%)|95% CI|', '|---|---|---:|---:|---:|---:|---|']
    for r in summary:lines.append(f"|{r['task']}|{r['method']}|{r['lr']:.6g}|{r['clip']}|{r['eps_scale']}|{r['accuracy_mean']:.3f} ± {r['accuracy_sd']:.3f}|[{r['accuracy_ci95_low']:.3f}, {r['accuracy_ci95_high']:.3f}]|")
    lines+=['','配对差定义为右方法减左方法；Student-t 95% 区间使用十个逐种子差，未经多重比较校正。全部方法两两差见 paired_differences.csv。不同选定超参数下的比较反映整个调参协议，不能直接归因为单一机制。',
        '', '训练效用：utility.csv、summary.csv。矩阵误差：matrix_errors.csv。敏感度与统一 GDP：sensitivity.csv。运行时间和内存：runtime_memory.csv；CPU 为训练父进程峰值 RSS，GPU 为 PyTorch allocated/reserved 峰值，DTC O(pT) 创新历史位于主机 FP32 RAM。',
        '', '矩阵目标严格使用指定 A；它是线性 workload 上的显式噪声误差，不是自适应 Adam 的有噪/无噪参数轨迹差。求解器收敛状态、迭代上限、障碍系数与候选轨迹完整保存在矩阵 JSON。',
        '', '只对明确的搜索/复核非有限值排除，未改变失败超参数自动重试。各阶段计划数、完成数和排除原因见 final_audit.json。']
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':report()
