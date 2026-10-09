"""Paired length comparison and conservative length selection."""
from dataclasses import replace
import json
import numpy as np
from scipy.stats import t
from exp8a import RESULTS
from exp8a.config import LENGTHS, SEEDS, save_json
from exp8a.benchmark_batch import isolated, write_csv
from exp8a.train import run

ACCURACY_FIELDS = ('max_length','seed','geometry','accuracy','loss','seconds','epsilon','mu',
                   'physical_batch_size','initialization_sha256','classifier_initialization_sha256',
                   'checkpoint_sha256','split_sha256','lr','C','eps_scale')


def benchmark_lengths(cfg,gpu):
    rows=[]
    for length in LENGTHS:
        for geometry in ('standard','scale'):
            rows.append(isolated(replace(cfg,max_length=length),geometry,gpu,category='length_benchmarks'))
            write_csv(RESULTS/'max_length_benchmark.csv',rows)
    return rows


def paired_statistics(rows):
    by_length={n:{r['seed']:r['accuracy'] for r in rows if r['max_length']==n} for n in LENGTHS}
    seeds=sorted(by_length[64])
    assert seeds and all(sorted(by_length[n])==seeds for n in LENGTHS)
    means={n:float(np.mean(list(by_length[n].values()))) for n in LENGTHS}
    best=max(LENGTHS,key=lambda n:means[n])
    stats={}
    for length in LENGTHS:
        differences=np.array([by_length[best][s]-by_length[length][s] for s in seeds])
        sem=float(differences.std(ddof=1)/np.sqrt(len(seeds))) if len(seeds)>1 else None
        mean=float(differences.mean())
        ci=[mean-float(t.ppf(.975,len(seeds)-1))*sem,mean+float(t.ppf(.975,len(seeds)-1))*sem] if sem is not None else None
        stats[str(length)]=dict(mean_accuracy=means[length],mean_gap_to_best=mean,
                               paired_gap_ci95=ci,paired_seeds=seeds)
    return best,stats


def choose_length(rows,benchmarks,tolerance=.003,baseline_accuracy=None):
    best,stats=paired_statistics(rows)
    safe={n:all(r['status']=='completed' and r['headroom_fraction']>=.2 for r in benchmarks if r['max_length']==n)
          and len([r for r in benchmarks if r['max_length']==n])==2 for n in LENGTHS}
    if baseline_accuracy is not None and max(v['mean_accuracy'] for v in stats.values())<=baseline_accuracy+tolerance:
        assert safe[64]
        return 64,dict(reason='No length exceeds the majority baseline by 0.003; equal baseline accuracy is insufficient length evidence; prefer 64',
                       statistics=stats,tolerance=tolerance,baseline_accuracy=baseline_accuracy,convergence_evidence=False)
    enough=len(stats['64']['paired_seeds'])>=3
    if not enough:
        assert safe[64], 'Fallback length 64 does not have required VRAM headroom'
        return 64,dict(reason='Single paired seed is insufficient; prefer 64',statistics=stats,tolerance=tolerance)
    eligible=[n for n in LENGTHS if safe[n] and stats[str(n)]['mean_gap_to_best']<=tolerance
              and stats[str(n)]['paired_gap_ci95'][1]<=tolerance]
    if not eligible:
        assert safe[64]
        return 64,dict(reason='Paired uncertainty exceeds 0.3 percentage points; prefer 64',statistics=stats,tolerance=tolerance)
    chosen=min(eligible)
    return chosen,dict(reason='Shortest safe length with mean and paired 95% upper gap <= 0.003',
                       statistics=stats,tolerance=tolerance,best_mean_length=best)


def train_lengths(cfg,gpu,benchmarks):
    rows=[]
    for length in LENGTHS:
        result=run(replace(cfg,max_length=length,seed=SEEDS[0]),gpu)
        rows.append(result)
        write_csv(RESULTS/'max_length_accuracy.csv',rows,ACCURACY_FIELDS)
    means=[r['accuracy'] for r in rows]
    # If any lengths are close, expand the entire paired design to three seeds (9 runs).
    near=any(abs(means[i]-means[j])<=.006 for i in range(3) for j in range(i))
    if near:
        for seed in SEEDS[1:]:
            for length in LENGTHS:
                rows.append(run(replace(cfg,max_length=length,seed=seed),gpu))
                write_csv(RESULTS/'max_length_accuracy.csv',rows,ACCURACY_FIELDS)
    assert len(rows) in (3,9)
    return summarize_lengths(rows,benchmarks)


def summarize_lengths(rows,benchmarks):
    token_stats=json.loads((RESULTS/'token_lengths.json').read_text())
    counts=token_stats['search_validation_label_counts']
    baseline=max(counts)/sum(counts)
    length,reason=choose_length(rows,benchmarks,baseline_accuracy=baseline)
    compute={str(n):dict(combined_steps_per_second=2/sum(1/r['logical_steps_per_second'] for r in benchmarks if r['max_length']==n),
                        peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in benchmarks if r['max_length']==n),
                        truncation_ratio=token_stats['truncation_ratio'][str(n)]) for n in LENGTHS}
    reason.update(compute_and_truncation=compute,full_runs=len(rows),close_candidate_trigger=.006,
                  token_lengths_file='exp8a/results/token_lengths.json')
    save_json(RESULTS/'convergence_diagnostics.json',dict(majority_baseline_accuracy=baseline,
              best_mean_accuracy=max(v['mean_accuracy'] for v in reason['statistics'].values()),
              convergence_evidence=reason.get('convergence_evidence',True),full_runs=len(rows),
              final_losses=[dict(seed=r['seed'],max_length=r['max_length'],loss=r['loss']) for r in rows],
              no_additional_pilots_after_freezing=True))
    save_json(RESULTS/'length_selection.json',dict(max_length=length,**reason))
    return length,rows,reason
