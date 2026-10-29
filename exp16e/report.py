"""Ten-seed descriptive comparisons, without controlled experiments."""
import itertools,statistics,math
from scipy.stats import t
from exp16e import RESULTS,ROOT
from exp16e.config import FINAL_SEEDS,METHODS,save_json,file_hash
from exp16e.support import read,write_csv,code_hashes

def stats(values):
    values=list(values);n=len(values);mean=statistics.mean(values);sd=statistics.stdev(values);half=float(t.ppf(.975,n-1))*sd/math.sqrt(n)
    return dict(n=n,mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half)

def generate(formal,freeze):
    matrix=read(RESULTS/'matrix_audit.json');per=[];summary=[];pairs=[]
    for task in ('cv','nlp'):
        for method in METHODS:
            rows=sorted([r for r in formal if r['task']==task and r['method']==method],key=lambda r:r['seed'])
            assert [r['seed'] for r in rows]==list(FINAL_SEEDS)
            v=freeze['winners'][task][method]
            for r in rows:
                assert r['status']=='completed' and r['official_evaluation_used'] and all(r[k]==v[k] for k in ('lr','clip','eps_scale'))
                per.append(dict(task=task,method=method,seed=r['seed'],lr=r['lr'],clip=r['clip'],eps_scale=r['eps_scale'],accuracy=r['accuracy'],loss=r['validation_loss'],clip_fraction=statistics.mean(e['clip_fraction'] for e in r['epochs']),final_clip_fraction=r['epochs'][-1]['clip_fraction'],epsilon=r['actual_epsilon'],delta=1e-5,seconds=r['seconds'],train_seconds=r['train_seconds'],peak_allocated_bytes=r['peak_allocated_bytes'],peak_reserved_bytes=r['peak_reserved_bytes'],source=r['source'],historical=r['historical'],summary_sha256=file_hash(ROOT/r['source']/'summary.json')))
            result=dict(task=task,method=method,lr=v['lr'],clip=v['clip'],eps_scale=v['eps_scale'])
            for field,out in [('accuracy','accuracy'),('validation_loss','loss')]:result.update({f'{out}_{k}':x for k,x in stats(r[field] for r in rows).items()})
            result.update(clip_fraction_mean=statistics.mean(p['clip_fraction'] for p in per if p['task']==task and p['method']==method),seconds_mean=statistics.mean(r['seconds'] for r in rows),peak_allocated_bytes_max=max(r['peak_allocated_bytes'] for r in rows),historical_runs=sum(r['historical'] for r in rows),epsilon=8,delta=1e-5)
            summary.append(result)
        for left,right in itertools.combinations(METHODS,2):
            a={r['seed']:r for r in formal if r['task']==task and r['method']==left};b={r['seed']:r for r in formal if r['task']==task and r['method']==right}
            # Pair only equal non-noise random inputs, not equal hyperparameters.
            for seed in FINAL_SEEDS:
                for k in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace'):
                    assert a[seed][k]==b[seed][k],(task,seed,k,left,right)
            pairs.append(dict(task=task,left=left,right=right,**{'accuracy_difference_'+k:v for k,v in stats(b[s]['accuracy']-a[s]['accuracy'] for s in FINAL_SEEDS).items()},**{'loss_difference_'+k:v for k,v in stats(b[s]['validation_loss']-a[s]['validation_loss'] for s in FINAL_SEEDS).items()}))
    write_csv(RESULTS/'formal_per_seed.csv',per)
    for task in ('cv','nlp'):write_csv(RESULTS/f'{task}_summary.csv',[r for r in summary if r['task']==task])
    write_csv(RESULTS/'paired_differences.csv',pairs)
    text=['# exp16e 最终报告','', '状态：全部必要正式训练、统计和审计已完成。每方法10种子；仅IID与B²LT，各方法独立选参。', '', '|任务|方法|lr|clip|eps_scale|准确率均值 ± SD (%)|95% CI (%)|loss均值|复用种子数|','|---|---|---:|---:|---:|---:|---|---:|---:|']
    for r in summary:text.append(f"|{r['task']}|{r['method']}|{r['lr']:.12g}|{r['clip']}|{r['eps_scale']}|{100*r['accuracy_mean']:.3f} ± {100*r['accuracy_sd']:.3f}|[{100*r['accuracy_ci95_low']:.3f}, {100*r['accuracy_ci95_high']:.3f}]|{r['loss_mean']:.6f}|{r['historical_runs']}|")
    text+=['','逐种子准确率、损失、裁剪比例、耗时、显存、隐私预算及复用路径见 formal_per_seed.csv。均值/标准差/Student-t 95%区间见 cv_summary.csv 与 nlp_summary.csv；同正式种子右减左差及区间见 paired_differences.csv，未作多重比较校正。','', '选参仅使用三种子内部验证准确率均值，交叉熵用于并列排序；竞争力判据、全部候选和复核证据见 selection_evidence.json、search_results.json。未新增或输出固定超参数受控实验结果。不同超参数下的算法比较不能解释为单一机制的因果效应。历史官方评价已使用，不声称新盲测。','', '矩阵逆/分解误差、敏感度、线性workload误差见 matrix_audit.json。当前clip的实际理论噪声误差见 theoretical_errors.csv；该误差不是自适应优化器参数轨迹误差。','', '单次DP训练 ε=8、δ=1e-5，固定五轮参与、GDP校准、不使用采样放大。搜索、复核和重复训练不具有联合ε=8保证。内部验证沿用历史公开验证假设，裁剪与梯度诊断仅作内部研究审计。']
    from exp16e.history import config
    errors=[]
    for task,methods in freeze['winners'].items():
        for method,v in methods.items():
            c=config(v);h=310 if task=='nlp' else 250
            a=next(a for a in matrix if a['total_steps']==h and a['workload']==c.workload and a['mechanism']==c.mechanism)
            errors.append(dict(task=task,method=method,clip=c.clip,lr=c.lr,matrix_sha256=a['sha256'],inverse_max_abs=a['inverse_max_abs'],factorization_max_abs=a['factorization_max_abs'],sensitivity=a['sensitivity'],objective=a['objective'],innovation_std_sum=c.clip*a['sigma_sum_clip1'],gradient_noise_MSE=c.clip**2*a['mean_gradient_noise_MSE_clip1'],workload_noise_MSE=c.clip**2*a['workload_noise_MSE_clip1']))
    write_csv(RESULTS/'theoretical_errors.csv',errors)
    (RESULTS/'final_report.md').write_text('\n'.join(text)+'\n')
    audit=dict(status='completed',formal_count=len(formal),historical_count=sum(r['historical'] for r in formal),new_count=sum(not r['historical'] for r in formal),seeds=list(FINAL_SEEDS),methods=list(METHODS),gpus=[1,2,3],nlp_max_concurrent=2,controlled_experiments=False,code_sha256=code_hashes(),budget=read(RESULTS/'budget.json'))
    outputs=['search_results.json','selection_evidence.json','final_freeze.json','cv_summary.csv','nlp_summary.csv','paired_differences.csv','formal_per_seed.csv','final_report.md','theoretical_errors.csv']
    audit['output_sha256']={n:file_hash(RESULTS/n) for n in outputs};save_json(RESULTS/'final_audit.json',audit)
