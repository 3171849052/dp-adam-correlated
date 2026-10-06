"""Final utilities, diagnostic metadata and seed-paired factorial comparisons."""
from exp6.runtime import EXP, require_curve
import json
from pathlib import Path
import numpy as np
from exp6.config import METHODS
from exp6.launch_batch import read_completed
from exp6.audit import diagnostics


def aggregate(values):
    assert len(values) >= 2
    return dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)),values=values)


def comparisons(values):
    effects = dict(MF_gain_raw=aggregate([b-a for a,b,c,d in values]),
                   MF_gain_scale=aggregate([d-c for a,b,c,d in values]),
                   scale_gain_iid=aggregate([c-a for a,b,c,d in values]),
                   scale_gain_mf=aggregate([d-b for a,b,c,d in values]),
                   interaction=aggregate([d-c-b+a for a,b,c,d in values]))
    return effects


def report(folder=EXP / 'results/final'):
    manifest = json.loads((folder / 'manifest.json').read_text())
    summaries = [read_completed(job) for job in manifest['jobs']]
    table = {(s['trial']['seed'],s['trial']['method']):s for s in summaries}
    seeds = manifest['seeds']
    for seed in seeds:
        paired = [table[seed,m] for m in METHODS]
        for key in ('initialization_sha256','order_sha256'):
            assert len({s[key] for s in paired}) == 1
        for key in ('augmentation_sha256','innovation_sha256'):
            assert all(s[key] == paired[0][key] for s in paired)
    metrics = {}
    for m in METHODS:
        per_seed = []
        for seed in seeds:
            s = table[seed,m]
            job = next(j for j in manifest['jobs'] if j['trial']['method'] == m and j['trial']['seed'] == seed)
            config = json.loads((Path(job['result_dir']) / 'config.json').read_text())
            per_seed.append(dict(seed=seed,final_test_top1=s['final_test_top1'],trial_id=s['trial_id'],
                fingerprint=s['fingerprint'],privacy=s['final_privacy'],calibration=s['calibration'],
                coefficients=config['coefficients'],diagnostics=diagnostics(job['result_dir'])))
        frozen = manifest['selected']['methods'][m]
        metrics[m] = dict(**aggregate([r['final_test_top1'] for r in per_seed]),per_seed=per_seed,
                          selected_hyperparameters=frozen['hyperparameters'],workload=frozen['workload'])
    values = [[table[s,m]['final_test_top1'] for m in METHODS] for s in seeds]
    effects = comparisons(values)
    result = dict(seeds=seeds,selected=manifest['selected'],methods=metrics,paired_effects=effects,
        research_question=dict(MF_gain_scale_gt_MF_gain_raw=effects['interaction']['mean'] > 0,
             difference=effects['interaction'],per_seed=[v > 0 for v in effects['interaction']['values']],
             interpretation='descriptive 3-seed paired comparison; each method independently utility-tuned'))
    (folder / 'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    text = ['# Exp6 final utility report','', '| Method | mean top1 | sample std | C | lr | geom_eps |','|---|---:|---:|---:|---:|---:|']
    for m,r in metrics.items():
        hp=r['selected_hyperparameters']
        text.append(f'| {m} | {r["mean"]:.6f} | {r["std"]:.6f} | {hp["C"]:g} | {hp["lr"]:g} | {hp["geom_eps"] if m.endswith("-scale") else "inactive"} |')
    text += ['', '| Seed | IID | MF | IID-scale | MF-scale |','|---|---:|---:|---:|---:|']
    text += [f'| {seed} | '+ ' | '.join(f'{v:.6f}' for v in row)+' |' for seed,row in zip(seeds,values)]
    text += ['', 'Seed-paired effects (top1 fractions; sample std, ddof=1):','']
    text += [f'- {k}: {r["mean"]:.6f} ± {r["std"]:.6f}; per seed {r["values"]}' for k,r in effects.items()]
    text += ['',f'MF_gain_scale > MF_gain_raw: {result["research_question"]["MF_gain_scale_gt_MF_gain_raw"]} (descriptive).',
             '', 'Full privacy, workload coefficients, clipping and LoRA geometry diagnostics are in report.json.']
    (folder / 'report.md').write_text('\n'.join(text)+'\n')
    return result


if __name__ == '__main__':
    require_curve()
    report()
