"""Full numeric report and exported figures; never influences hyperparameters."""
import csv
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import t
from exp12 import ROOT, RESULTS
from exp12.config import METHOD, file_hash, save_json


def write_csv(path, rows):
    rows=list(rows)
    if not rows: return
    fields=[key for key in rows[0] if all(not isinstance(r.get(key),(dict,list,tuple)) for r in rows)]
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def stats(values):
    x=np.asarray(values,dtype=float);assert len(x)>=2
    sd=float(x.std(ddof=1));half=float(t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)))
    return dict(n=len(x),mean=float(x.mean()),std=sd,ci95=[float(x.mean()-half),float(x.mean()+half)],
                method='Two-sided Student t interval, sample standard deviation ddof=1')


def md_table(rows,columns):
    if not rows: return '尚无结果。\n'
    def value(x):
        if isinstance(x,float): return f'{x:.8g}'
        if isinstance(x,(list,dict)): return json.dumps(x,ensure_ascii=False)
        return str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
           ['| '+' | '.join(value(r.get(k,'')) for k in columns)+' |' for r in rows])+'\n'


def generate():
    all_rows=[]
    for stage in ('search','recheck','final'):
        all_rows += [json.loads(p.read_text()) for p in sorted((RESULTS/stage).glob('cv/*/summary.json'))]
    completed=[r for r in all_rows if r['status']=='completed']
    grid=[r for r in completed if r['stage']=='search'];rechecks=[r for r in completed if r['stage']=='recheck']
    finals=sorted([r for r in completed if r['stage']=='final'],key=lambda r:r['seed'])
    failure=json.loads((RESULTS/'failure.json').read_text()) if (RESULTS/'failure.json').exists() else None
    completion=json.loads((RESULTS/'completion.json').read_text()) if (RESULTS/'completion.json').exists() else None
    status='completed' if completion else 'failed' if failure else 'running'
    text=['# Exp12：CV BC2-Naive BandInvMF\n',
          f'状态：**{status}**。完整训练 {len(completed)}/34（网格 {len(grid)}/20，复核 {len(rechecks)}/4，正式 {len(finals)}/10）。\n',
          '仅将 Exp9 的累计一阶动量偏差校正 workload 乘上逐步 `sqrt(1-beta2^s)`；省略全局 `(1-beta1)`。'
          'DP-Adam、全参数微调、增强、裁剪、噪声生成和 GDP 校准保持 Exp9。'
          '四带 D=[1,d1,d2,d3]；目标 Sens(D^-1)^2 ||WD||_F^2/T。'
          '使用 Exp9 两个固定起点的 Nelder–Mead（maxiter=1600, xatol=1e-9, fatol=1e-7），不宣称全局最优。\n',
          '数据完全离线：CIFAR-100 现有训练/测试文件和 ViT-Tiny 预训练参数；内部 45000/5000 分层划分 seed=20261008。'
          '每次训练五轮，逻辑批量1000、物理批量250，Adam betas=(0.9,0.999), eps=1e-8, weight_decay=0。'
          '每种子固定初始化及一次排列，五轮重复此排列；噪声 RNG seed+1；DataLoader 增强 RNG 与 Exp9 一致。\n',
          '## 矩阵与 workload\n']
    matrices=[]
    for n in (225,250):
        path=RESULTS/f'matrices/T{n}/optimization.json'
        if path.exists():
            meta=json.loads(path.read_text())
            for label in ('old_filter','new_filter'):
                matrices.append(dict(T=n,filter=label,**meta[label]))
    text.append(md_table(matrices,['T','filter','coefficients','sensitivity','old_workload_error','new_workload_error','old_objective','new_objective']))
    text.append('误差为 ||WD||_F^2/T；两种 workload 的数值尺度不同，优化收益应在同一列内比较新旧滤波器。'
                'T=225 用于网格与复核，T=250 用于全量正式训练，两套矩阵在测试集评价前冻结。\n')
    write_csv(RESULTS/'matrix_comparison.csv',[dict(r,coefficients=json.dumps(r['coefficients'])) for r in matrices])
    text.extend(['## 20组网格结果\n','LR∈{0.001,0.003,0.005,0.007}，C∈{3,10,30,100,200}；seed=20261101。'
                 '仅按第五轮内部验证准确率降序、canonical trial ID 升序选 Top-2。\n',
                 md_table(sorted(grid,key=lambda r:(r['lr'],r['C'])),['lr','C','seed','accuracy','seconds','trial_id']),
                 '## Top-2复核与冻结\n',md_table(sorted(rechecks,key=lambda r:(r['lr'],r['C'],r['seed'])),['lr','C','seed','accuracy','seconds','trial_id'])])
    if (RESULTS/'selection_evidence.json').exists():
        selection=json.loads((RESULTS/'selection_evidence.json').read_text())
        text.append(md_table(selection,['lr','C','mean_validation_accuracy','base_id']))
        winner=json.loads((RESULTS/'frozen_configs.json').read_text())
        text.append(f'冻结最优 LR={winner["lr"]}、C={winner["C"]}；三种子均值={winner["mean_validation_accuracy"]:.8f}。'
                    '按三个种子的第五轮准确率均值选择，平局按原搜索 trial ID；测试结果不参与选择。'
                    f' frozen_manifest SHA-256：`{file_hash(RESULTS/"frozen_manifest.json")}`。\n')
    text.extend(['## 10种子正式结果\n',md_table(finals,['seed','accuracy','actual_epsilon','seconds','physical_gpu','peak_allocated_bytes'])])
    summary={}
    if len(finals)==10:
        summary['formal']=stats([r['accuracy'] for r in finals])
        s=summary['formal'];text.append(f'准确率均值 **{100*s["mean"]:.4f}%**，样本标准差 **{100*s["std"]:.4f} pp**；'
                           f'95% Student-t CI **[{100*s["ci95"][0]:.4f}%, {100*s["ci95"][1]:.4f}%]**（df=9）。\n')
    elif finals:
        text.append('正式实验未全部完成，不输出10种子统计。\n')
    text.append('## 与Exp9 Momentum-Bias的同种子配对\n')
    audit_path=RESULTS/'baseline_audit.json';pairs=[]
    if audit_path.exists():
        baseline=json.loads(audit_path.read_text());text.append(f'历史基线审计状态：{baseline["status"]}。\n')
        if baseline['status']=='passed':
            text.append(f'Exp9 正式冻结超参数 LR={baseline["lr"]}、C={baseline["C"]}。'
                        '已检查10种子的原始 summary、训练配置、源码、矩阵、检查点及日志指标文件哈希，'
                        '并重建原始 workload/滤波器和校准。进一步逐种子检查初始化、分类头、数据顺序和增强 RNG 轨迹。\n')
            if finals:
                from exp12.audit import pairing
                old=[r for r in json.loads((ROOT/'exp9/results/final_results.json').read_text()) if r['task']=='cv' and r['method']==METHOD]
                for r in old:
                    evidence=next(e for e in baseline['runs'] if e['seed']==r['seed'])
                    assert file_hash(ROOT/'exp9/results'/r['result_dir']/'summary.json')==evidence['summary_sha256']
                pairs=pairing(finals,old)
                save_json(RESULTS/'paired_differences.json',pairs);write_csv(RESULTS/'paired_differences.csv',pairs)
                text.append(md_table(pairs,['seed','exp12_accuracy','exp9_accuracy','pairable','difference','mismatches']))
                if len(pairs)==10 and all(r['pairable'] for r in pairs):
                    summary['paired']=stats([r['difference'] for r in pairs]);s=summary['paired']
                    text.append(f'Exp12−Exp9 配对差均值 **{100*s["mean"]:+.4f} pp**；95% Student-t CI '
                                f'**[{100*s["ci95"][0]:+.4f}, {100*s["ci95"][1]:+.4f}] pp**（df=9）。\n')
                else:
                    text.append('完整有效配对不足10个，不输出全体配对置信区间；不一致历史记录不会作为可配对基线。\n')
        else:
            text.append('基线不一致，禁止输出配对差值及其置信区间。原因：'+baseline['reason']+'\n')
    text.append('## 隐私校准审计\n')
    privacy=[dict(stage=r['stage'],seed=r['seed'],lr=r['lr'],C=r['C'],T=r['optimizer_steps'],
                  sensitivity=r['calibration']['sensitivity'],mu=r['calibration']['target_mu'],
                  innovation_std_sum=r['calibration']['innovation_std_sum'],epsilon=r['actual_epsilon'],delta=1e-5,
                  adjacency=r['calibration']['adjacency'],sampling_amplification=False) for r in completed]
    write_csv(RESULTS/'privacy_audit.csv',privacy)
    text.append(md_table(privacy,['stage','seed','lr','C','T','sensitivity','mu','innovation_std_sum','epsilon']))
    text.append('固定五轮参与模式（间隔45/50）的绝对 Gram 敏感度界，add/remove zero-out 邻接；无抽样放大。'
                'GDP delta(mu,8)=1e-5；innovation_std_sum=C*Sens/mu；逻辑梯度除以1000。'
                '每轮使用完整参与日程的前缀敏感度审计 ε≤8，完整训练校验 ε=8。'
                'ε=8 是每次训练的预算；整个34次搜索/选择/重复运行流程没有 ε=8 的组合声明，内部验证按公开数据处理。\n')
    text.append('## 运行、GPU和可复现审计\n')
    if completion:
        timezone=ZoneInfo('Asia/Shanghai')
        text.append('开始：'+datetime.fromtimestamp(completion['started_unix'],timezone).isoformat()+'；结束：'+
                    datetime.fromtimestamp(completion['finished_unix'],timezone).isoformat()+
                    f'；总墙钟 {completion["seconds"]:.2f} 秒；保护目录未改变：{completion["protected_files_unchanged"]}。\n')
        text.append('GPU调度审计：`'+json.dumps(completion['scheduler'],ensure_ascii=False)+'`。\n')
    if completed:
        computation=[]
        for gpu in sorted({r['physical_gpu'] for r in completed}):
            runs=[r for r in completed if r['physical_gpu']==gpu]
            computation.append(dict(gpu=gpu,trainings=len(runs),worker_seconds=sum(r['seconds'] for r in runs),
                     training_seconds=sum(r['train_seconds'] for r in runs),
                     peak_allocated_GiB=max(r['peak_allocated_bytes'] for r in runs)/2**30,
                     peak_reserved_GiB=max(r['peak_reserved_bytes'] for r in runs)/2**30))
        write_csv(RESULTS/'compute_summary.csv',computation)
        text.append(md_table(computation,['gpu','trainings','worker_seconds','training_seconds','peak_allocated_GiB','peak_reserved_GiB']))
    if (RESULTS/'gpu_telemetry.jsonl').exists():
        samples={}
        for line in (RESULTS/'gpu_telemetry.jsonl').read_text().splitlines():
            e=json.loads(line)
            for gpu_line in e['raw'].splitlines():
                g,u,m=map(float,gpu_line.split(','));samples.setdefault(int(g),[]).append((u,m))
        telemetry=[dict(gpu=g,samples=len(v),sample_mean_utilization_percent=float(np.mean([x[0] for x in v])),
                        peak_memory_MiB=max(x[1] for x in v)) for g,v in sorted(samples.items())]
        write_csv(RESULTS/'gpu_utilization.csv',telemetry)
        text.append(md_table(telemetry,['gpu','samples','sample_mean_utilization_percent','peak_memory_MiB']))
        text.append('GPU利用率为每约5秒遥测的样本均值，原始记录保存在 gpu_telemetry.jsonl；每卡最多一个CV进程。\n')
    text.append('源码 SHA-256 随配置记录；冻结清单绑定24次选择证据、资产和两套矩阵；每次 summary 绑定配置、'
                '检查点、矩阵、排列和训练指标哈希。result_manifest.json 绑定报告、结果、日志和图表；'
                'workflow.log 在退出后单独记录哈希。所有新增文件位于 exp12/。\n')
    if failure:
        text.append('## 失败记录\n'+failure['error']+'\n\n```\n'+failure['traceback']+'```\n')
    summary.update(status=status,completed_trainings=len(completed),search=len(grid),recheck=len(rechecks),final=len(finals))
    save_json(RESULTS/'statistics.json',summary)
    (RESULTS/'report.md').write_text('\n'.join(text))
    if len(grid)==20:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4));lrs=(.001,.003,.005,.007);clips=(3,10,30,100,200)
        values=np.array([[next(r['accuracy'] for r in grid if r['lr']==lr and r['C']==c) for c in clips] for lr in lrs])
        image=ax.imshow(values*100,aspect='auto');ax.set_xticks(range(5),clips);ax.set_yticks(range(4),lrs)
        ax.set_xlabel('Clipping C');ax.set_ylabel('LR');ax.set_title('Exp12 fifth-epoch internal validation accuracy (%)')
        for i in range(4):
            for j in range(5): ax.text(j,i,f'{100*values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]<.4 else 'black')
        fig.colorbar(image,ax=ax);fig.tight_layout();fig.savefig(RESULTS/'grid.png',dpi=180);plt.close(fig)
        if len(pairs)==10 and all(r['pairable'] for r in pairs):
            fig,axs=plt.subplots(1,2,figsize=(10,4));seeds=[r['seed'] for r in pairs]
            axs[0].plot(seeds,[100*r['exp9_accuracy'] for r in pairs],'o-',label='Exp9 Momentum-Bias')
            axs[0].plot(seeds,[100*r['exp12_accuracy'] for r in pairs],'o-',label='Exp12 BC2-Naive')
            axs[0].set_ylabel('Test accuracy (%)');axs[0].legend()
            axs[1].bar(range(10),[100*r['difference'] for r in pairs]);axs[1].axhline(0,color='black',lw=1)
            axs[1].set_xticks(range(10),[str(s)[-2:] for s in seeds]);axs[1].set_xlabel('Seed suffix');axs[1].set_ylabel('Paired difference (pp)')
            fig.tight_layout();fig.savefig(RESULTS/'paired.png',dpi=180);plt.close(fig)


if __name__=='__main__': generate()
