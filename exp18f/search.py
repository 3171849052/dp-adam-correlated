"""Predeclared one-axis scan, adaptive LR refinement, and five-seed review."""
import math
from collections import defaultdict
from exp18f.common import (BASE,PARAMS,METHODS,BASELINE,SEARCH_SEEDS,TURN_SEEDS,REVIEW_SEEDS,
    read,save,sha,now,make_config,configs,outdir,terminal,calibration)

def axis_candidates():
    space=read(BASE/'search_space.json')
    return [(f'{axis}_{i:02d}',dict(BASELINE,**{axis:value}))
            for axis in PARAMS for i,value in enumerate(space['axes'][axis])]

def stage_plan(stage,candidates,seeds):
    path=BASE/'plans'/f'{stage}.json'
    jobs=[make_config(stage,cid,m,s,p) for cid,p in candidates for s in seeds for m in METHODS]
    if path.exists():assert read(path)==jobs
    else:save(path,jobs)
    save(BASE/'search'/f'{stage}_candidates.json',[dict(candidate=c,parameters=p) for c,p in candidates])
    save(BASE/'search'/f'{stage}_calibration.json',{c:{m:calibration(make_config(stage,c,m,seeds[0],p))
        for m in METHODS} for c,p in candidates})
    return jobs

def stages():return [p.stem for p in sorted((BASE/'plans').glob('*.json')) if p.stem!='smoke']

def aggregate(stage_names,required=True):
    import numpy as np
    grouped={};pairing={}
    for stage in stage_names:
        for c in configs(stage):
            r=terminal(c)
            if required:assert r is not None,f'Unfinished: {outdir(c)}'
            if r is None:continue
            if 'pairing' in r:
                if c['seed'] in pairing:assert r['pairing']==pairing[c['seed']]
                pairing[c['seed']]=r['pairing']
            row=grouped.setdefault(c['candidate'],dict(candidate=c['candidate'],parameters={p:c[p] for p in PARAMS},trials=[]))
            row['trials'].append(dict(method=c['method'],seed=c['seed'],status=r['status'],
                metrics=r.get('metrics'),directory=str(outdir(c).relative_to(BASE))))
    result=[]
    for row in grouped.values():
        ok=[t for t in row['trials'] if t['status']=='complete']
        row.update(complete=len(ok),failures=len(row['trials'])-len(ok),
            mean_accuracy=float(np.mean([t['metrics']['accuracy'] for t in ok])) if ok else None,
            mean_loss=float(np.mean([t['metrics']['loss'] for t in ok])) if ok else None)
        result.append(row)
    return sorted(result,key=lambda r:(r['failures']>0,-(r['mean_accuracy'] or 0),r['mean_loss'] or 1e30,r['candidate']))

def stability(rows,baselines):
    """Numeric failures are terminal; finite predictive deterioration is a separate label."""
    import numpy as np
    result=[]
    for r in sorted(rows,key=lambda r:r['parameters']['lr']):
        per_method={}
        for m in METHODS:
            trials=[t for t in r['trials'] if t['method']==m]
            good=[t for t in trials if t['status']=='complete'];base=baselines[m]
            acc=float(np.mean([t['metrics']['accuracy'] for t in good])) if good else None
            loss=float(np.mean([t['metrics']['loss'] for t in good])) if good else None
            numeric=len(good)!=len(trials)
            deteriorated=bool(good and (acc<base['accuracy']-.05 or loss>max(1.,1.5*base['loss'])))
            label='numerical_unstable' if numeric else ('predictive_unstable' if deteriorated else 'stable')
            per_method[m]=dict(label=label,mean_accuracy=acc,mean_loss=loss,n=len(trials))
        result.append(dict(candidate=r['candidate'],lr=r['parameters']['lr'],methods=per_method,
            stable=all(v['label']=='stable' for v in per_method.values()),mean_accuracy=r['mean_accuracy']))
    return result

def critical(points,method=None,numerical_only=False):
    def stable(p):
        if numerical_only:
            labels=[p['methods'][method]['label']] if method else [v['label'] for v in p['methods'].values()]
            return all(label!='numerical_unstable' for label in labels)
        return p['methods'][method]['label']=='stable' if method else p['stable']
    flags=[stable(p) for p in points]
    # Any return to stability invalidates monotone bisection, even below the peak.
    nonmonotonic=any(not flags[i] and flags[j] for i in range(len(flags)) for j in range(i+1,len(flags)))
    transitions=[dict(low=a['lr'],high=b['lr']) for a,b in zip(points,points[1:]) if stable(a) and not stable(b)]
    return dict(nonmonotonic=nonmonotonic,intervals=transitions,
        high_side_interval=transitions[-1] if transitions else None,
        coverage='bracketed' if transitions else ('right_censored' if all(flags) else 'no_stable_to_unstable_bracket'))

def refinement(points,used):
    info=critical(points)
    if info['nonmonotonic']:
        gaps=[(a,b) for a,b in zip(points,points[1:]) if a['stable']!=b['stable']]
        a,b=max(gaps,key=lambda ab:math.log(ab[1]['lr']/ab[0]['lr']))
        return math.sqrt(a['lr']*b['lr']),'local_densification_nonmonotonic'
    if info['intervals']:
        interval=info['intervals'][-1]
        return math.sqrt(interval['low']*interval['high']),'geometric_bisection'
    # No observed bracket: locally densify the largest performance drop; never invent a bracket.
    gaps=[(a,b) for a,b in zip(points,points[1:]) if math.sqrt(a['lr']*b['lr']) not in used]
    if gaps:
        a,b=max(gaps,key=lambda ab:((ab[0]['mean_accuracy'] or 0)-(ab[1]['mean_accuracy'] or 0),math.log(ab[1]['lr']/ab[0]['lr'])))
        return math.sqrt(a['lr']*b['lr']),'local_densification_unbracketed'
    return None,'exhausted'

def baseline_metrics(rows):
    import numpy as np
    baseline=next(r for r in rows if r['parameters']==BASELINE and r['candidate'].startswith('lr_'))
    result={}
    for m in METHODS:
        ok=[t for t in baseline['trials'] if t['method']==m and t['status']=='complete']
        result[m]=dict(accuracy=float(np.mean([t['metrics']['accuracy'] for t in ok])) if ok else .60,
            loss=float(np.mean([t['metrics']['loss'] for t in ok])) if ok else 2/3,
            successful_trials=len(ok),reference='observed baseline' if ok else 'predeclared absolute thresholds')
    return result

def update_turning(stage_names,baselines,history):
    rows=aggregate(stage_names)
    lr_rows=[r for r in rows if r['candidate'].startswith(('lr_','turn_'))]
    points=stability(lr_rows,baselines)
    best=next((r for r in rows if r['failures']==0),None)
    lr_best=next((r for r in sorted(lr_rows,key=lambda r:(r['failures']>0,-(r['mean_accuracy'] or 0))) if r['failures']==0),None)
    peaks_by_method={}
    for method in METHODS:
        candidates=[]
        for r in lr_rows:
            trials=[t for t in r['trials'] if t['method']==method]
            if trials and all(t['status']=='complete' for t in trials):
                candidates.append(dict(lr=r['parameters']['lr'],candidate=r['candidate'],
                    mean_accuracy=sum(t['metrics']['accuracy'] for t in trials)/len(trials)))
        peaks_by_method[method]=max(candidates,key=lambda r:r['mean_accuracy']) if candidates else None
    value=dict(status='in_progress',time=now(),baseline=baselines,points=points,history=history,
        common=critical(points),per_method={m:critical(points,m) for m in METHODS},
        numerical_only=dict(common=critical(points,numerical_only=True),
            per_method={m:critical(points,m,numerical_only=True) for m in METHODS}),
        performance_peak=lr_best,performance_peak_by_method=peaks_by_method,raw_best=best,
        note='Stability labels use predeclared internal metrics. Alpha decay alone is never failure. Unbracketed bounds remain censored.')
    save(BASE/'turning_point.json',value);return points,value

def run_search():
    from exp18f.scheduler import run
    run('axis');rows=aggregate(['axis']);save(BASE/'search/axis_ranking.json',rows)
    baselines=baseline_metrics(rows);history=[];names=['axis']
    points,turn=update_turning(names,baselines,history)
    for i in range(3):
        lr,mode=refinement(points,{p['lr'] for p in points})
        if lr is None:break
        stage=f'turn{i+1}';stage_plan(stage,[(f'turn_{i+1}',dict(BASELINE,lr=lr))],TURN_SEEDS)
        history.append(dict(round=i+1,lr=lr,mode=mode,pre_search=critical(points)))
        run(stage);names.append(stage);points,turn=update_turning(names,baselines,history)
    turn['status']='complete';save(BASE/'turning_point.json',turn)
    # All three preferred rounded LR values are reviewed, using entirely new paired seeds.
    rounded=[(f'rounded_{i+1}',dict(BASELINE,lr=lr)) for i,lr in enumerate(read(BASE/'search_space.json')['rounded_lrs'])]
    stage_plan('rounded',rounded,REVIEW_SEEDS);run('rounded')
    reviewed=aggregate(['rounded']);eligible=[r for r in reviewed if r['complete']==10 and r['failures']==0]
    save(BASE/'search/rounded_ranking.json',reviewed)
    frozen=dict(status='frozen' if eligible else 'no_eligible_candidate',frozen=bool(eligible),time=now(),
        parameters=eligible[0]['parameters'] if eligible else None,ranking=reviewed,
        selection='ONLY five new paired seeds; pooled final internal accuracy descending, then loss ascending; complete ten-run candidates only',
        seeds=REVIEW_SEEDS,official_validation_used=False,
        review_plan_sha256=sha(BASE/'plans/rounded.json'))
    save(BASE/'rounded_freeze.json',frozen)
    assert sum(len(configs(s)) for s in stages())<=156
    return frozen
