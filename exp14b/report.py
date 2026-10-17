import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exp14b import ROOT, RESULTS
from exp14b.support import METHODS, BASE_LR, FACTORS, EPS, write_csv

def main():
    search=json.loads((RESULTS/'search_results.json').read_text())
    final=json.loads((RESULTS/'confirmation_results.json').read_text())
    historical={}
    for exp in ('exp9','exp14a'):
        historical[exp]=[json.loads(p.read_text()) for p in (ROOT/exp/'results/final/cv').glob('*/summary.json')]
    comparison=[]
    for r in final:
        row=dict(method=r['method'],seed=r['seed'],lr=r['lr'],eps=r['eps'],exp14b_accuracy=r['accuracy'])
        for exp in historical:
            ref=next(s for s in historical[exp] if s['method']==r['method'] and s['seed']==r['seed'] and s['epsilon']==8)
            assert ref['status']=='completed'
            for key in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','strategy_sha256','workload_sha256','rng_trace'):
                assert ref[key]==r[key],(exp,key)
            row[exp+'_accuracy']=ref['accuracy'];row['difference_vs_'+exp+'_pp']=100*(r['accuracy']-ref['accuracy'])
        comparison.append(row)
    write_csv(RESULTS/'comparison.csv',comparison)
    write_csv(RESULTS/'confirmation_results.csv',[{k:r[k] for k in ('method','lr','eps','accuracy','optimizer_steps','seconds')} for r in final])
    write_csv(RESULTS/'search_results.csv',[dict(method=r['method'],number=r['number'],lr=r['lr'],eps=r['eps'],accuracy=r['accuracy'],validation_loss=r['epochs'][-1]['validation_loss']) for r in search])
    fig,axes=plt.subplots(1,4,figsize=(18,4))
    effects=[]
    for ax,method in zip(axes,METHODS):
        cells=[r for r in search if r['method']==method]
        z=np.array([[next(r['accuracy'] for r in cells if r['lr']==BASE_LR[method]*f and r['eps']==e)*100 for e in EPS] for f in FACTORS])
        im=ax.imshow(z);ax.set_xticks(range(3),[str(e) for e in EPS]);ax.set_yticks(range(3),[str(BASE_LR[method]*f) for f in FACTORS]);ax.set_xlabel('eps');ax.set_ylabel('lr');ax.set_title(method.replace('dp-adam-',''))
        for i in range(3):
            for j in range(3): ax.text(j,i,f'{z[i,j]:.2f}',ha='center',va='center',color='white')
        fig.colorbar(im,ax=ax);effects.append(f'- {method}: 固定学习率时，三个 eps 的验证准确率极差分别为 '+', '.join(f'{v:.2f}' for v in np.ptp(z,axis=1))+' 个百分点。')
    fig.tight_layout();fig.savefig(RESULTS/'lr_eps_heatmaps.png');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(18,8))
    for j,method in enumerate(METHODS):
        for r in [s for s in search if s['method']==method]+[s for s in final if s['method']==method]:
            label=f"{r['stage']} lr={r['lr']:g} eps={r['eps']:g}"
            for ax,key in zip(axes[:,j],('accuracy','validation_loss')):
                ax.plot([e['epoch'] for e in r['epochs']],[e[key] for e in r['epochs']],label=label);ax.set_ylabel(key);ax.set_xlabel('epoch')
        axes[0,j].set_title(method.replace('dp-adam-',''));axes[1,j].legend(fontsize=5)
    fig.tight_layout();fig.savefig(RESULTS/'training_curves.png');plt.close(fig)
    lines=['# Exp14b：MF + AdaM³ 学习率与稳定项搜索','', '36 次内部验证搜索与 4 次完整训练确认均完成。种子 20261111，C=30，epsilon=8，delta=1e-5。仅按最后一轮内部验证准确率、验证损失、固定编号依次选参；官方测试集未用于选参。', '', '|机制|最优 lr|最优 eps|Exp9 测试 %|Exp14a 测试 %|Exp14b 测试 %|相对 Exp14a 百分点|','|---|---:|---:|---:|---:|---:|---:|']
    for r in comparison:
        lines.append(f"|{r['method']}|{r['lr']:g}|{r['eps']:g}|{r['exp9_accuracy']*100:.2f}|{r['exp14a_accuracy']*100:.2f}|{r['exp14b_accuracy']*100:.2f}|{r['difference_vs_exp14a_pp']:+.2f}|")
    lines+=['','缩小学习率并搜索 eps 后，'+str(sum(r['difference_vs_exp14a_pp']>0 for r in comparison))+'/4 种机制的确认准确率高于 Exp14a。该比较反映联合选参结果，不能单独归因于学习率。','',*effects,'','eps 的可见影响用上述固定学习率下的极差描述；单种子无法判定统计显著性或稳定的跨种子影响。','', 'AdaM³ 直接复用 Exp14a：v 输入为未校正 m 的平方，分别做偏差校正，eps 位于根号外，无额外学习率补偿。初始化、样本顺序、增强随机流、MF 矩阵与历史确认结果逐项核验。隐私预算按单次训练校准，不宣称全部搜索/选参过程合成为 epsilon=8。']
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
